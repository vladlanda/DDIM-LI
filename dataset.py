"""
Dataset for METSAT lightning nowcasting.

File format: {id}_{start}_{end}_{channel}.jpg + .wld
Channels:
  - ir  : IR 105 (always present)
  - li  : lightning index (always present)
  - ch0 : BT 105 alias or other (optional)
  - ch1 : BT 123 (optional)
  - ch2 : BT 87  (optional)
  - chN : up to ch9 (optional)

A "sample" is a temporal sequence of T_in consecutive timesteps
followed by T_out target timesteps.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Ensure dataset log output goes through tqdm.write() so progress bars are
# not corrupted. train.py installs the handler on the root logger; this call
# is a no-op if that handler is already present (i.e. when called from train).
def _ensure_tqdm_logging():
    from tqdm import tqdm as _tqdm
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and
           type(h).__name__ == "_TqdmLoggingHandler" for h in root.handlers):
        return  # already set up by train.py
    class _H(logging.StreamHandler):
        def emit(self, record):
            try:
                _tqdm.write(self.format(record))
            except Exception:
                self.handleError(record)
    handler = _H()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

_ensure_tqdm_logging()

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
REQUIRED_CHANNELS = ["ir", "li"]
OPTIONAL_CHANNELS = [f"ch{i}" for i in range(10)]
ALL_CHANNELS = REQUIRED_CHANNELS + OPTIONAL_CHANNELS  # ir, li, ch0..ch9

# Channel indices in the tensor (only channels present in dataset)
CHANNEL_FILL_VALUES = {
    "ir": 0.0,    # will be normalised anyway
    "li": 0.0,    # sparse – missing = no lightning
    **{f"ch{i}": 0.0 for i in range(10)},
}


# -------------------------------------------------------------------
# Statistics (computed once, stored as JSON next to the dataset)
# -------------------------------------------------------------------
def compute_or_load_stats(
    root: str,
    channel_list: List[str],
    stat_path: Optional[str] = None,
    n_samples: int = 2000,
) -> Dict[str, Dict[str, float]]:
    """Return per-channel mean/std (or median/iqr for LI).
    Results are saved to stat_path so they are only computed once.
    """
    if stat_path and os.path.exists(stat_path):
        logger.info(f"  Loading channel stats from cache: {stat_path}")
        with open(stat_path) as f:
            return json.load(f)

    logger.info(f"  Computing channel statistics (first run — will be cached) …")
    accum     = defaultdict(list)
    all_files = list(Path(root).rglob("*.jpg"))
    np.random.shuffle(all_files)
    sample    = all_files[:n_samples]

    for fp in tqdm(sample, desc="  Computing stats", unit="img",
                   dynamic_ncols=True, leave=True):
        ch = fp.stem.split("_")[-1]
        if ch not in channel_list:
            continue
        img = np.array(Image.open(fp).convert("L"), dtype=np.float32) / 255.0
        accum[ch].append(img.ravel())

    stats = {}
    for ch, arrays in accum.items():
        flat = np.concatenate(arrays)
        if ch == "li":
            flat = np.cbrt(flat)
            stats[ch] = {"mean": float(flat.mean()), "std": float(flat.std() + 1e-6),
                         "transform": "cbrt"}
        else:
            stats[ch] = {"mean": float(flat.mean()), "std": float(flat.std() + 1e-6),
                         "transform": "linear"}

    if stat_path:
        with open(stat_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Channel stats cached → {stat_path}")
    return stats


def normalize(img: np.ndarray, stats: Dict, ch: str) -> np.ndarray:
    if stats[ch]["transform"] == "cbrt":
        img = np.cbrt(img)
    return (img - stats[ch]["mean"]) / stats[ch]["std"]


def denormalize(img: np.ndarray, stats: Dict, ch: str) -> np.ndarray:
    img = img * stats[ch]["std"] + stats[ch]["mean"]
    if stats[ch]["transform"] == "cbrt":
        img = np.power(img, 3)
    return img


# -------------------------------------------------------------------
# Timestamp parsing  (IR-only, fast path)
# -------------------------------------------------------------------
TS_FMT = "%Y%m%dT%H%M%SZ"

def parse_ir_filename(path: Path) -> Optional[datetime]:
    """
    Fast path: parse only the start timestamp from an *_ir.jpg filename.
    Format: {id}_{start}_{end}_ir.jpg
    Returns start datetime or None.
    """
    name  = path.stem                    # strip .jpg
    parts = name.rsplit("_", 3)
    if len(parts) != 4:
        return None
    _, start_s, _, ch = parts
    if ch != "ir":
        return None
    try:
        return datetime.strptime(start_s, TS_FMT)
    except ValueError:
        return None


def stem_to_prefix(stem: str) -> str:
    """
    Strip the channel suffix from a file stem to get the shared prefix.
    e.g. '10895_20250130T211007Z_20250130T211924Z_ir' → '10895_20250130T211007Z_20250130T211924Z'
    """
    return stem.rsplit("_", 1)[0]


# -------------------------------------------------------------------
# Index builder  (with disk cache to avoid slow repeat filesystem scans)
# -------------------------------------------------------------------
def build_index(root: str,
                cache_path: Optional[str] = None) -> Dict[datetime, Dict[str, Path]]:
    """
    Returns: { start_dt: { channel: Path, ... }, ... }
    Only timesteps that have BOTH ir AND li are included.

    Speed improvements vs previous version:
      - Uses glob (flat) instead of rglob (recursive) — your files are flat per region
      - Scans only *_ir.jpg files (anchor channel) — 1/C of total files
      - Other channel paths are derived by string substitution — zero extra stat calls
      - Results cached to JSON so the scan only runs once per dataset folder
    """
    if cache_path is None:
        cache_path = str(Path(root) / ".index_cache.json")

    # --- Load from cache ---
    if os.path.exists(cache_path):
        logger.info(f"  Loading index cache: {Path(cache_path).name}")
        with open(cache_path) as f:
            raw_json = json.load(f)
        index = {
            datetime.fromisoformat(dt_s): {ch: Path(p) for ch, p in chs.items()}
            for dt_s, chs in raw_json.items()
        }
        logger.info(f"  Index loaded: {len(index)} valid timesteps")
        return index

    # --- Build from scratch: scan IR files only ---
    logger.info(f"  Building index for {Path(root).name} (first run — will be cached)")

    ir_files = sorted(Path(root).glob("*_ir.jpg"))   # flat glob, IR only
    index: Dict[datetime, Dict[str, Path]] = {}

    for ir_fp in tqdm(ir_files, desc=f"  Scanning {Path(root).name}",
                      unit="file", dynamic_ncols=True, leave=True):

        start_dt = parse_ir_filename(ir_fp)
        if start_dt is None:
            continue

        prefix = ir_fp.parent / stem_to_prefix(ir_fp.stem)

        # Derive all channel paths from the shared prefix — no extra stat calls
        chs: Dict[str, Path] = {}
        for ch in ALL_CHANNELS:
            candidate = Path(str(prefix) + f"_{ch}.jpg")
            if candidate.exists():
                chs[ch] = candidate

        # Only include timesteps that have both required channels
        if all(c in chs for c in REQUIRED_CHANNELS):
            index[start_dt] = chs

    # --- Save cache ---
    # NOTE: Caching is disabled — uncomment if filesystem scanning is slow
    # (e.g. USB/network drives). On a local SSD the scan is fast enough.
    # cache_data = {
    #     dt.isoformat(): {ch: str(p) for ch, p in chs.items()}
    #     for dt, chs in index.items()
    # }
    # with open(cache_path, "w") as f:
    #     json.dump(cache_data, f)
    # logger.info(f"  Index cached → {cache_path}  ({len(index)} valid timesteps)")

    logger.info(f"  Index built: {len(index)} valid timesteps")

    return index


# -------------------------------------------------------------------
# Valid sequence builder  (sliding window, single sorted pass)
# -------------------------------------------------------------------
def build_valid_starts(
    sorted_times: List[datetime],
    dt:           timedelta,
    seq_len:      int,
    root_name:    str = "",
) -> List[List[datetime]]:
    """
    Returns a list of valid sequences, where each sequence is a list of
    seq_len actual datetime objects taken directly from sorted_times.

    Storing the full sequence (not just t0) is critical because file
    timestamps have per-file jitter (e.g. 10:20:03 instead of 10:20:00).
    Recomputing times as t0 + i*dt would generate datetimes that don't
    exist in the index → KeyError in __getitem__.
    """
    if len(sorted_times) < seq_len:
        return []

    dt_seconds = int(dt.total_seconds())
    gap_min    = dt_seconds - 60    # allow ±1 min slack
    gap_max    = dt_seconds + 60

    # Convert to integer seconds-since-epoch for fast gap arithmetic
    epochs = [int(t.timestamp()) for t in sorted_times]
    N      = len(epochs)
    valid  = []

    for i in tqdm(range(N - seq_len + 1),
                  desc=f"  Validating {root_name}",
                  unit="ts", dynamic_ncols=True, leave=True):

        ok = True
        for j in range(i, i + seq_len - 1):
            gap = epochs[j + 1] - epochs[j]
            if not (gap_min <= gap <= gap_max):
                ok = False
                break
        if ok:
            # Store the actual datetime objects — never recompute from t0
            valid.append(sorted_times[i : i + seq_len])

    return valid


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class METSATDataset(Dataset):
    """
    Each item is:
        context : (T_in,  C, H, W)  – normalised input frames
        target  : (T_out, C, H, W)  – normalised target residuals
        lead_times: (T_out,)         – lead time in multiples of dt_min
        mask    : (C,)               – 1 if channel present, 0 if filled

    Residual target = target_frame - context_frame[-1]
    """

    def __init__(
        self,
        root: str,
        channel_list: List[str],
        T_in: int = 6,
        T_out: int = 36,
        dt_min: int = 10,
        img_size: Tuple[int,int] = (256, 256),
        stats: Optional[Dict] = None,
        stat_path: Optional[str] = None,
        augment: bool = True,
        max_samples: Optional[int] = None,
        binary_li_ctx: bool = True,
        ctx_channels: Optional[List[str]] = None,
    ):
        self.root         = root
        self.channel_list = channel_list
        self.C            = len(channel_list)
        self.T_in         = T_in
        self.T_out        = T_out
        self.dt           = timedelta(minutes=dt_min)
        self.dt_min       = dt_min
        self.img_size     = img_size
        self.augment      = augment
        self.max_samples  = max_samples

        # Build temporal index
        self.index = build_index(root)
        self.sorted_times = sorted(self.index.keys())

        # Stats
        self.stats = stats or compute_or_load_stats(
            root, channel_list, stat_path=stat_path
        )

        # Build valid sequences (each is a list of seq_len actual datetimes)
        seq_len = T_in + T_out
        self.valid_sequences = build_valid_starts(
            self.sorted_times, self.dt, seq_len,
            root_name=Path(root).name,
        )

        if max_samples is not None and max_samples < len(self.valid_sequences):
            step = len(self.valid_sequences) // max_samples
            self.valid_sequences = self.valid_sequences[::step][:max_samples]
            logger.info(f"Dataset '{root}': {len(self.valid_sequences)} sequences "
                        f"(capped at max_samples={max_samples})")
        else:
            logger.info(f"Dataset '{root}': {len(self.valid_sequences)} valid sequences")

        # Pre-compute normalisation arrays for vectorised apply in _load_frame
        self._norm_mean = np.array([
            self.stats[ch]["mean"] if ch in self.stats else 0.5
            for ch in channel_list], dtype=np.float32)
        self._norm_std  = np.array([
            self.stats[ch]["std"]  if ch in self.stats else 0.5
            for ch in channel_list], dtype=np.float32)
        self.binary_li_ctx  = binary_li_ctx
        self.ctx_channels   = ctx_channels
        # ctx_idx: indices of channels to include in context
        if ctx_channels is not None:
            self.ctx_idx = [channel_list.index(c) for c in ctx_channels
                            if c in channel_list]
        else:
            self.ctx_idx = list(range(len(channel_list)))
        self._cbrt_mask = np.array([
            self.stats[ch]["transform"] == "cbrt" if ch in self.stats else False
            for ch in channel_list])

    def __len__(self):
        return len(self.valid_sequences)

    def _load_frame(self, dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (C, H, W) normalised frame and (C,) mask.
        """
        chs  = self.index[dt]
        h, w = self.img_size
        frame = np.zeros((self.C, h, w), dtype=np.float32)
        mask  = np.zeros(self.C,        dtype=np.float32)

        for i, ch in enumerate(self.channel_list):
            if ch not in chs:
                frame[i] = CHANNEL_FILL_VALUES.get(ch, 0.0)
                continue
            img = Image.open(chs[ch]).convert("L")
            if img.size != (w, h):
                img = img.resize((w, h), Image.BILINEAR)
            arr = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(h, w)
            frame[i] = arr.astype(np.float32) * (1.0 / 255.0)
            mask[i]  = 1.0

        for i in range(self.C):
            if mask[i] == 0.0:
                continue
            if self._cbrt_mask[i]:
                frame[i] = np.cbrt(frame[i])
            frame[i] = (frame[i] - self._norm_mean[i]) / self._norm_std[i]

        return frame, mask

    def __getitem__(self, idx):
        # Use the pre-validated actual datetimes — no arithmetic, no jitter
        times = self.valid_sequences[idx]   # List[datetime], length T_in + T_out

        frames, masks = zip(*[self._load_frame(t) for t in times])
        frames = np.stack(frames)   # (T_in+T_out, C, H, W)
        masks  = np.stack(masks)    # (T_in+T_out, C)

        context        = frames[:self.T_in]
        target_abs     = frames[self.T_in:]
        last_ctx       = context[-1:]
        target_residual = target_abs - last_ctx

        lead_times = np.arange(1, self.T_out + 1, dtype=np.float32)

        if self.augment and np.random.rand() > 0.5:
            context         = context[:, :, :, ::-1].copy()
            target_residual = target_residual[:, :, :, ::-1].copy()

        # LI density: fraction of non-zero LI pixels across all target frames.
        # Used for dynamic li_weight during training (Cui et al. 2019).
        li_idx = self.channel_list.index("li") if "li" in self.channel_list else None
        if li_idx is not None:
            # target_abs in normalised space: zero cbrt(LI) normalised = -mean/std
            # but we want physical > 0, so use target_abs before residual:
            li_abs    = target_abs[:, li_idx]          # (T_out, H, W) normalised abs
            li_phys   = li_abs * self._norm_std[li_idx] + self._norm_mean[li_idx]
            if self._cbrt_mask[li_idx]:
                li_phys = np.power(np.clip(li_phys, 0.0, None), 3)
            li_density = float((li_phys > 0).mean())  # fraction of non-zero pixels
        else:
            li_density = 0.0

        # Apply ctx_channels selection — subset context to requested channels only.
        # If ctx_channels=["ir","ch0","ch1"] then LI is excluded from context,
        # forcing the model to predict LI purely from cloud structure.
        ctx = context[:, self.ctx_idx, :, :]   # (T_in, C_ctx_sel, H, W)

        # Binary LI context: append (li >= 1/255) as extra channel per context frame.
        # Only appended if LI is included in ctx_channels.
        # Following Ravuri et al. 2021 (DGMR) explicit rain mask conditioning.
        _li_in_ctx = (li_idx is not None and li_idx in self.ctx_idx)
        if self.binary_li_ctx and _li_in_ctx:
            _li_ctx_pos = self.ctx_idx.index(li_idx)
            li_norm = ctx[:, _li_ctx_pos]                             # (T_in, H, W)
            li_phys = li_norm * self._norm_std[li_idx] + self._norm_mean[li_idx]
            if self._cbrt_mask[li_idx]:
                li_phys = np.power(np.clip(li_phys, 0.0, None), 3)
            li_bin  = (li_phys >= 1.0 / 255.0).astype(np.float32)    # (T_in, H, W)
            context_out = np.concatenate(
                [ctx, li_bin[:, None, :, :]], axis=1                  # (T_in, C_ctx+1, H, W)
            )
        else:
            context_out = ctx

        return {
            "context":    torch.from_numpy(context_out),
            "target":     torch.from_numpy(target_residual),
            "lead_times": torch.from_numpy(lead_times),
            "ctx_mask":   torch.from_numpy(masks[:self.T_in]),
            "tgt_mask":   torch.from_numpy(masks[self.T_in:]),
            "last_ctx":   torch.from_numpy(last_ctx[0]),
            "li_density": torch.tensor(li_density, dtype=torch.float32),
        }


# -------------------------------------------------------------------
# Multi-region concatenated dataset
# -------------------------------------------------------------------
class MultiRegionDataset(Dataset):
    def __init__(self, roots: List[str], **kwargs):
        self.datasets = [METSATDataset(r, **kwargs) for r in roots]
        self.lengths  = [len(d) for d in self.datasets]
        self.cumlen   = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumlen[-1]

    def __getitem__(self, idx):
        ds_idx = np.searchsorted(self.cumlen[1:], idx, side="right")
        local  = idx - self.cumlen[ds_idx]
        return self.datasets[ds_idx][local]


# -------------------------------------------------------------------
# DataModule helpers
# -------------------------------------------------------------------

def compute_li_sample_weights(
    dataset,
    oversample_factor: float = 5.0,
    density_percentile: float = 75.0,
) -> np.ndarray:
    """
    Compute per-sequence sampling weights for WeightedRandomSampler.

    Context: the LI channel in this dataset is a continuous lightning index
    that is non-zero in ~96% of frames (central Africa has very high lightning
    frequency). Binary active/inactive splitting is therefore not meaningful —
    virtually all sequences are "active". The real imbalance is at the pixel
    level (~6% non-zero pixels per frame), which is handled by li_weight.

    Instead, we stratify by lightning DENSITY: sequences whose mean LI
    pixel fraction across target frames exceeds the `density_percentile`
    threshold are oversampled by `oversample_factor`. This ensures the model
    sees proportionally more high-density convective events, which are the
    hardest and most important cases to predict correctly.

    Per-sequence mean LI density is computed by loading one representative
    target frame per sequence (the middle target frame) and measuring the
    fraction of pixels with value > 0. Loading all T_out frames per sequence
    would be more accurate but prohibitively slow for 52k sequences.

    Args:
        dataset            : METSATDataset instance
        oversample_factor  : weight for high-density sequences (default 5.0)
        density_percentile : sequences above this percentile of LI density
                             are considered high-density (default 75th pct)

    Returns:
        np.ndarray of shape (len(dataset),) with per-sequence weights
    """
    ch_list = getattr(dataset, "channel_list", None)
    if ch_list is None or "li" not in ch_list:
        logger.warning("compute_li_sample_weights: no LI channel, returning uniform weights")
        return np.ones(len(dataset), dtype=np.float64)

    h, w = dataset.img_size

    # ── Pass 1: compute mean LI density per sequence (one frame per seq) ────
    densities = np.zeros(len(dataset), dtype=np.float32)

    for i, seq_times in enumerate(tqdm(
        dataset.valid_sequences,
        desc  = "  Computing LI densities",
        unit  = "seq",
        leave = False,
        dynamic_ncols = True,
    )):
        target_times = seq_times[dataset.T_in:]   # T_out target timestamps
        # Use the middle target frame as representative
        mid_t = target_times[len(target_times) // 2]

        if mid_t not in dataset.index:
            continue
        ch_files = dataset.index[mid_t]
        if "li" not in ch_files:
            continue

        try:
            img = Image.open(ch_files["li"]).convert("L")
            if img.size != (w, h):
                img = img.resize((w, h), Image.BILINEAR)
            arr = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(h, w)
            densities[i] = float((arr > 0).mean())
        except Exception:
            densities[i] = 0.0

    # ── Pass 2: assign weights based on density percentile ──────────────────
    threshold   = float(np.percentile(densities, density_percentile))
    weights     = np.where(densities >= threshold,
                           float(oversample_factor), 1.0).astype(np.float64)

    n_high = int((densities >= threshold).sum())
    n_low  = len(dataset) - n_high
    logger.info(
        f"  LI density threshold (p{density_percentile:.0f}): {threshold:.4f} "
        f"({threshold*100:.2f}% of pixels)"
    )
    logger.info(
        f"  High-density sequences: {n_high} (×{oversample_factor:.1f}), "
        f"low-density: {n_low} (×1.0)"
    )
    return weights


def make_dataloaders(
    train_roots:       List[str],
    channel_list:      List[str],
    T_in:              int   = 6,
    T_out:             int   = 36,
    img_size:          Tuple[int, int] = (256, 256),
    batch_size:        int   = 4,
    num_workers:       int   = 4,
    stat_path:         Optional[str] = None,
    max_samples:       Optional[int] = None,
    train_val_split:   float = 0.7,
    oversample_factor:  float = 5.0,
    density_percentile: float = 75.0,
    binary_li_ctx:      bool  = True,
    ctx_channels:       Optional[List[str]] = None,
):
    """
    Builds train and validation loaders from train_roots only.

    The split is done TEMPORALLY per region — the first `train_val_split`
    fraction of each region's sequences go to train, the rest to val.
    This is the correct approach for time-series data: you must never
    let future frames leak into the training set via random shuffling.

    test_roots (your held-out regions) are NOT touched here.
    Use make_test_loader() after training for final evaluation on those.
    """
    assert 0.0 < train_val_split < 1.0, "train_val_split must be in (0, 1)"

    # Build each region dataset once, then slice valid_sequences in-place.
    # No second construction pass — avoids duplicate index scanning and logs.
    full_ds = MultiRegionDataset(
        train_roots, channel_list=channel_list,
        T_in=T_in, T_out=T_out, img_size=img_size,
        stat_path=stat_path, augment=False,
        max_samples=max_samples,
    )
    shared_stats = full_ds.datasets[0].stats

    import copy

    train_parts, val_parts = [], []

    for ds in full_ds.datasets:
        all_seqs = ds.valid_sequences          # full list, not yet mutated
        n        = len(all_seqs)
        n_train  = max(1, int(n * train_val_split))

        train_seqs = all_seqs[:n_train]        # slice BEFORE any mutation
        val_seqs   = all_seqs[n_train:]

        # Train part: reuse the existing dataset object
        train_ds = ds
        train_ds.valid_sequences = train_seqs
        train_ds.augment         = True
        train_parts.append(train_ds)

        # Val part: shallow-copy (shares index/stats, no re-scan)
        val_ds = copy.copy(ds)
        val_ds.valid_sequences = val_seqs
        val_ds.augment         = False
        val_parts.append(val_ds)

        logger.info(
            f"  {ds.root}: {n} sequences → "
            f"{len(train_seqs)} train / {len(val_seqs)} val  "
            f"(split={train_val_split:.0%})"
        )

    class _ConcatDS(Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.lengths  = [len(d) for d in datasets]
            self.cumlen   = np.cumsum([0] + self.lengths)
        def __len__(self):
            return int(self.cumlen[-1])
        def __getitem__(self, idx):
            ds_idx = int(np.searchsorted(self.cumlen[1:], idx, side="right"))
            return self.datasets[ds_idx][idx - self.cumlen[ds_idx]]

    train_combined = _ConcatDS(train_parts)
    val_combined   = _ConcatDS(val_parts)

    logger.info(
        f"Total — train: {len(train_combined)} sequences, "
        f"val: {len(val_combined)} sequences"
    )

    # Stratified sequence sampling: oversample lightning-active sequences.
    # Uses WeightedRandomSampler which is compatible with DDP (train.py
    # replaces it with DistributedSampler when running multi-GPU).
    from torch.utils.data import WeightedRandomSampler

    # Compute weights per sequence across all training parts
    all_weights = np.concatenate([
        compute_li_sample_weights(ds,
                                   oversample_factor  = oversample_factor,
                                   density_percentile = density_percentile)
        for ds in train_parts
    ])
    train_sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(all_weights).double(),
        num_samples = len(train_combined),
        replacement = True,
    )

    train_loader = DataLoader(
        train_combined, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_combined, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, shared_stats


def make_test_loader(
    test_roots:   List[str],
    channel_list: List[str],
    stats:        Dict,
    T_in:         int   = 6,
    T_out:        int   = 36,
    img_size:     Tuple[int, int] = (256, 256),
    batch_size:   int   = 4,
    num_workers:  int   = 4,
    max_samples:  Optional[int] = None,
):
    """
    Loader for the held-out test regions.
    Only call this after training is complete.
    Stats must be passed in from the training set — never refit on test data.

    Sequences are made fully non-overlapping by striding valid_sequences by
    (T_in + T_out).  This ensures that no two test sequences share any frames,
    giving independent CRPS/CSI samples and honest confidence intervals.
    """
    stride = T_in + T_out   # e.g. 6+36=42 — full sequence length

    test_ds = MultiRegionDataset(
        test_roots, channel_list=channel_list,
        T_in=T_in, T_out=T_out, img_size=img_size,
        stats=stats, augment=False,
        max_samples=max_samples,
    )

    # Apply non-overlapping stride per region dataset
    total_before = sum(len(ds.valid_sequences) for ds in test_ds.datasets)
    for ds in test_ds.datasets:
        ds.valid_sequences = ds.valid_sequences[::stride]
    total_after = sum(len(ds.valid_sequences) for ds in test_ds.datasets)

    logger.info(
        f"Test loader: {total_before} → {total_after} sequences "
        f"after non-overlapping stride={stride} (T_in={T_in} + T_out={T_out})"
    )

    # Rebuild _ConcatDS lengths after stride
    test_ds.lengths = [len(ds.valid_sequences) for ds in test_ds.datasets]
    test_ds.cumlen  = np.cumsum([0] + test_ds.lengths)

    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ===================================================================
# CLI: visualise all DataLoader outputs for one sequence
#
# python dataset.py /home/vladlanda/Workplace/LI-DATASETS/small/central_africa_4
# python dataset.py /home/vladlanda/Workplace/LI-DATASETS/small/central_africa_4 --idx 42
# python dataset.py /home/vladlanda/Workplace/LI-DATASETS/small/central_africa_4 --out seq.png
# ===================================================================

if __name__ == "__main__":
    import argparse
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    p = argparse.ArgumentParser(
        description="Visualise all DataLoader outputs for one sequence."
    )
    p.add_argument("root",        help="Dataset root directory")
    p.add_argument("--idx",       type=int,   default=0)
    p.add_argument("--channels",  nargs="+",  default=["ir", "li", "ch0", "ch1"])
    p.add_argument("--T_in",      type=int,   default=18)
    p.add_argument("--T_out",     type=int,   default=6)
    p.add_argument("--img_size",  nargs=2,    type=int, default=[64, 64])
    p.add_argument("--stat_path", default=None)
    p.add_argument("--out",       default=None, help="Also save to PNG")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ds = METSATDataset(
        root         = args.root,
        channel_list = args.channels,
        T_in         = args.T_in,
        T_out        = args.T_out,
        img_size     = tuple(args.img_size),
        stat_path    = args.stat_path,
        augment      = False,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No valid sequences found in {args.root}")

    idx    = args.idx % len(ds)
    sample = ds[idx]
    stats  = ds.stats
    C      = len(args.channels)

    # ── Unpack all DataLoader outputs ───────────────────────────────────
    ctx_norm  = sample["context"].numpy()    # (T_in,  C, H, W) normalised absolute
    tgt_res   = sample["target"].numpy()     # (T_out, C, H, W) normalised RESIDUALS
    last_ctx  = sample["last_ctx"].numpy()   # (C, H, W)
    ctx_mask  = sample["ctx_mask"].numpy()   # (T_in,  C)  1=present 0=missing
    tgt_mask  = sample["tgt_mask"].numpy()   # (T_out, C)
    lead_times = sample["lead_times"].numpy() # (T_out,)  [1,2,...,T_out]
    li_density = float(sample["li_density"])

    # Reconstruct absolute target frames from residuals
    tgt_abs  = tgt_res + last_ctx[None]      # (T_out, C, H, W)

    # Denormalise to physical units
    def to_phys(arr_nc, ch):
        """arr_nc: (N, H, W) normalised → physical"""
        x = arr_nc * stats[ch]["std"] + stats[ch]["mean"]
        if stats[ch].get("transform") == "cbrt":
            x = np.power(np.clip(x, 0, None), 3)
        return x

    ctx_phys = np.stack([to_phys(ctx_norm[:, ci], ch)
                         for ci, ch in enumerate(args.channels)], axis=1)
    tgt_phys = np.stack([to_phys(tgt_abs[:, ci],  ch)
                         for ci, ch in enumerate(args.channels)], axis=1)
    res_phys = np.stack([to_phys(tgt_res[:, ci],  ch)
                         for ci, ch in enumerate(args.channels)], axis=1)

    cmaps = ["hot" if ch == "li" else "gray" for ch in args.channels]

    # ── Figure layout ───────────────────────────────────────────────────
    # Section 1: context frames          C rows × T_in cols
    # Section 2: target absolute frames  C rows × T_out cols
    # Section 3: target residuals        C rows × T_out cols
    # Section 4: masks                   2 rows (ctx/tgt) × C cols
    # Section 5: scalars bar             li_density + lead_times

    T_in  = args.T_in
    T_out = args.T_out
    cw, rh = 1.3, 1.5   # cell width / row height inches

    fig = plt.figure(figsize=((T_in + T_out * 2 + 1) * cw,
                               (C * 3 + 3) * rh * 0.6),
                     facecolor="white")

    outer = gridspec.GridSpec(5, 1, figure=fig,
                              hspace=0.55,
                              height_ratios=[C, C, C, 2, 1])

    def section_grid(outer_idx, ncols, nrows=C):
        return gridspec.GridSpecFromSubplotSpec(
            nrows, ncols,
            subplot_spec=outer[outer_idx],
            wspace=0.04, hspace=0.08
        )

    def section_title(outer_idx, title, color):
        ax = fig.add_subplot(outer[outer_idx])
        ax.set_visible(False)
        fig.text(
            ax.get_position().x0, ax.get_position().y1 + 0.005,
            title, fontsize=9, fontweight="bold", color=color,
            transform=fig.transFigure, va="bottom"
        )

    ft = 6   # font size for tick labels / titles

    # ── Section 1: context ──────────────────────────────────────────────
    section_title(0, f"① context  (T_in={T_in})  shape: ({T_in}, {C}, H, W)  "
                     f"[plotted in PHYSICAL units — model receives (cbrt(x)-mean)/std]",
                     "#1a5ea8")
    gs1 = section_grid(0, T_in)
    for ci, (ch, cmap) in enumerate(zip(args.channels, cmaps)):
        d = ctx_phys[:, ci]
        vmin, vmax = d.min(), d.max()
        if vmin == vmax: vmax += 1e-6
        # Show normalised range in ylabel so it's clear what model sees
        norm_data = ctx_norm[:, ci]   # (T_in, H, W) normalised
        nmin, nmax = float(norm_data.min()), float(norm_data.max())
        for t in range(T_in):
            ax = fig.add_subplot(gs1[ci, t])
            ax.imshow(d[t], cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(
                    f"{ch}\nphys [{vmin:.2f},{vmax:.2f}]\nnorm [{nmin:.1f},{nmax:.1f}]",
                    fontsize=ft-1, rotation=0, labelpad=28, va="center"
                )
            if ci == 0:
                ax.set_title(f"ctx-{T_in-t}", fontsize=ft,
                             pad=2, backgroundcolor="#dce8f5")

    # ── Section 2: target absolute ──────────────────────────────────────
    section_title(1, f"② target absolute  (T_out={T_out})  shape: ({T_out}, {C}, H, W)  "
                     f"= residual + last_ctx  [physical units — for reference only]",
                     "#1a7a3c")
    gs2 = section_grid(1, T_out)
    for ci, (ch, cmap) in enumerate(zip(args.channels, cmaps)):
        d = tgt_phys[:, ci]
        vmin, vmax = d.min(), d.max()
        if vmin == vmax: vmax += 1e-6
        for t in range(T_out):
            ax = fig.add_subplot(gs2[ci, t])
            ax.imshow(d[t], cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(ch, fontsize=ft, rotation=0,
                              labelpad=20, va="center")
            if ci == 0:
                ax.set_title(f"+{int(lead_times[t]*10)}m",
                             fontsize=ft, pad=2,
                             backgroundcolor="#d4edda")

    # ── Section 3: target residuals ─────────────────────────────────────
    section_title(2, f"③ target residuals  (T_out={T_out})  shape: ({T_out}, {C}, H, W)  "
                     f"= target_abs - last_ctx  [physical units — model predicts NORMALISED residuals]",
                     "#8b3a00")
    gs3 = section_grid(2, T_out)
    for ci, (ch, cmap) in enumerate(zip(args.channels, cmaps)):
        d = res_phys[:, ci]
        # Symmetric colormap for residuals
        absmax = max(abs(d.min()), abs(d.max()), 1e-6)
        for t in range(T_out):
            ax = fig.add_subplot(gs3[ci, t])
            ax.imshow(d[t], cmap="RdBu_r", vmin=-absmax, vmax=absmax,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(ch, fontsize=ft, rotation=0,
                              labelpad=20, va="center")
            if ci == 0:
                ax.set_title(f"+{int(lead_times[t]*10)}m",
                             fontsize=ft, pad=2,
                             backgroundcolor="#fde8d0")

    # ── Section 4: masks ────────────────────────────────────────────────
    gs4 = gridspec.GridSpecFromSubplotSpec(
        2, C, subplot_spec=outer[3], wspace=0.3, hspace=0.4
    )
    mask_labels = [
        (ctx_mask.T,  f"ctx_mask  ({T_in}, {C})"),
        (tgt_mask.T,  f"tgt_mask  ({T_out}, {C})")
    ]
    for row, (mask_data, title) in enumerate(mask_labels):
        ax = fig.add_subplot(gs4[row, :])
        ax.imshow(mask_data, cmap="Blues", vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=ft)
        ax.set_yticks(range(C))
        ax.set_yticklabels(args.channels, fontsize=ft)
        ax.set_xlabel("timestep", fontsize=ft)
        ax.tick_params(labelsize=ft)

    # ── Section 5: scalar outputs ────────────────────────────────────────
    ax5 = fig.add_subplot(outer[4])
    ax5.axis("off")

    ld_pct  = f"{li_density*100:.2f}"
    lt_list = lead_times.astype(int).tolist()
    lt_min  = (lead_times * 10).astype(int).tolist()
    lc_shp  = list(last_ctx.shape)
    info = (
        f"li_density = {li_density:.4f}  ({ld_pct}% of LI pixels > 0 across target frames)\n"
        f"lead_times = {lt_list}  (step indices -> x10 min = {lt_min} min ahead)\n"
        f"last_ctx shape = {lc_shp}  (last context frame, added back to residuals at eval)"
    )
    ax5.text(0.01, 0.85, info, transform=ax5.transAxes,
             fontsize=ft+1, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                       edgecolor="#cccccc", linewidth=0.8))

    fig.suptitle(
        f"DataLoader outputs — sequence {idx}/{len(ds)-1}  |  "
        f"{args.root.rstrip('/').split('/')[-1]}",
        fontsize=10, fontweight="bold", y=1.002
    )

    if args.out:
        plt.savefig(args.out, dpi=130, bbox_inches="tight",
                    facecolor="white")
        logger.info(f"Saved → {args.out}")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
