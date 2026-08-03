"""
Fast data loading from memmap-packed regions (see preprocess_to_memmap.py).

Deliberately kept as a FULLY SEPARATE path from dataset.py -- does not
modify the existing, validated JPEG-based pipeline in any way. Mirrors
METSATDataset/__getitem__ as closely as possible (same variable names,
same two-stage structure) specifically to make correctness review easy:
a reviewer should be able to compare this side-by-side with dataset.py's
_load_frame/__getitem__ and see they do the same thing.

Reuses build_valid_starts and compute_or_load_stats directly from
dataset.py rather than reimplementing sequence-validity or normalisation-
stats logic -- these are unchanged by how the raw pixels are stored.

CRITICAL correctness detail replicated exactly: _load_frame's normalisation
loop SKIPS cbrt/z-score for missing channels (mask==0), leaving them at
the raw, un-normalised CHANNEL_FILL_VALUES. This is NOT the same as
normalising a stored fill value -- it must be replicated as an explicit
skip, which is why missing channels are handled as a special case below
rather than by baking a fill value into the packed array.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    build_valid_starts, compute_or_load_stats, CHANNEL_FILL_VALUES
)

logger = logging.getLogger(__name__)


class PackedMETSATDataset(Dataset):
    def __init__(
        self,
        packed_dir: str,          # output of preprocess_to_memmap.py
        channel_list: List[str],
        T_in: int = 6,
        T_out: int = 36,
        dt_min: int = 10,
        stats: Optional[Dict] = None,
        stat_path: Optional[str] = None,
        stats_root: Optional[str] = None,   # original JPEG root, for stats
        augment: bool = True,
        max_samples: Optional[int] = None,
        binary_li_ctx: bool = True,
        ctx_channels: Optional[List[str]] = None,
    ):
        packed_dir = Path(packed_dir)
        with open(packed_dir / "meta.json") as f:
            meta = json.load(f)

        packed_channels = meta["channels"]
        missing = [c for c in channel_list if c not in packed_channels]
        if missing:
            raise ValueError(
                f"channel_list {channel_list} requests channels {missing} "
                f"not present in the packed data (packed with channels="
                f"{packed_channels}). Re-run preprocess_to_memmap.py with "
                f"all needed channels."
            )
        h, w = meta["img_size"]

        self.root         = str(packed_dir)   # for logging parity
        self.channel_list = channel_list
        self.C             = len(channel_list)
        self._packed_ch_idx = [packed_channels.index(c) for c in channel_list]
        self.T_in          = T_in
        self.T_out         = T_out
        self.dt            = timedelta(minutes=dt_min)
        self.dt_min        = dt_min
        self.img_size      = (h, w)
        self.augment       = augment
        self.max_samples   = max_samples

        N = meta["N"]
        self._frames = np.memmap(packed_dir / "frames.dat", dtype=np.uint8,
                                 mode="r", shape=(N, len(packed_channels), h, w))
        self._mask   = np.memmap(packed_dir / "mask.dat", dtype=np.uint8,
                                 mode="r", shape=(N, len(packed_channels)))

        self.sorted_times = [datetime.fromisoformat(t) for t in meta["timestamps"]]
        self._row_of = {t: i for i, t in enumerate(self.sorted_times)}

        # Stats: IDENTICAL source of truth as the JPEG-based pipeline --
        # computed from the ORIGINAL root's JPEGs (or a cached stats.json),
        # not derived from the packed data. Normalisation is a property of
        # the data distribution, independent of storage format.
        self.stats = stats or compute_or_load_stats(
            stats_root or str(packed_dir.parent), channel_list, stat_path=stat_path
        )

        seq_len = T_in + T_out
        self.valid_sequences = build_valid_starts(
            self.sorted_times, self.dt, seq_len,
            root_name=packed_dir.parent.name,
        )
        if max_samples is not None and max_samples < len(self.valid_sequences):
            step = len(self.valid_sequences) // max_samples
            self.valid_sequences = self.valid_sequences[::step][:max_samples]
        logger.info(f"PackedDataset '{packed_dir}': {len(self.valid_sequences)} "
                    f"valid sequences")

        self._norm_mean = np.array([
            self.stats[ch]["mean"] if ch in self.stats else 0.5
            for ch in channel_list], dtype=np.float32)
        self._norm_std  = np.array([
            self.stats[ch]["std"]  if ch in self.stats else 0.5
            for ch in channel_list], dtype=np.float32)
        self.binary_li_ctx = binary_li_ctx
        self.ctx_channels  = ctx_channels
        if ctx_channels is not None:
            _missing = [c for c in ctx_channels if c not in channel_list]
            if _missing:
                raise ValueError(f"ctx_channels {_missing} not in channel_list "
                                 f"{channel_list}.")
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
        Same output contract as METSATDataset._load_frame: (C,H,W) normalised
        frame + (C,) mask. Reads from the memmap instead of decoding JPEGs.
        """
        row = self._row_of[dt]
        h, w = self.img_size
        frame = np.zeros((self.C, h, w), dtype=np.float32)
        mask  = np.zeros(self.C, dtype=np.float32)

        # Plain scalar indexing (fast view), NOT arr[row, fancy_list] which
        # triggers NumPy's advanced indexing and was measured ~41x slower
        # on a memmap (see git history). Channel selection happens on the
        # small, already in-memory per-row result instead.
        row_frame = self._frames[row]
        row_mask  = self._mask[row]
        raw_frame = row_frame[self._packed_ch_idx]
        raw_mask  = row_mask[self._packed_ch_idx]

        for i, ch in enumerate(self.channel_list):
            if raw_mask[i] == 0:
                frame[i] = CHANNEL_FILL_VALUES.get(ch, 0.0)
                continue
            frame[i] = raw_frame[i].astype(np.float32) * (1.0 / 255.0)
            mask[i] = 1.0

        for i in range(self.C):
            if mask[i] == 0.0:
                continue
            if self._cbrt_mask[i]:
                frame[i] = np.cbrt(frame[i])
            frame[i] = (frame[i] - self._norm_mean[i]) / self._norm_std[i]

        return frame, mask

    def __getitem__(self, idx):
        # Identical to METSATDataset.__getitem__ from here on -- see
        # dataset.py for the reference implementation this must match.
        times = self.valid_sequences[idx]

        # NOTE: a "batched, single contiguous slice read" version of this
        # was tried and REVERTED. It was expected to help (sequential bulk
        # read vs many small reads is normally a big win for real disk
        # I/O), but rigorous interleaved, repeated benchmarking on a 0.39GB
        # realistic-scale file showed it was ~1.4x SLOWER than this
        # per-timestep loop, not faster -- reason not fully understood
        # (possibly numpy overhead in the boolean-mask vectorised
        # normalisation, possibly something about how memmap slice-to-array
        # materialisation behaves vs many small scalar-indexed reads that
        # benefit from OS readahead). Recorded honestly rather than shipped
        # on the assumption that batching must help.
        frames, masks = zip(*[self._load_frame(t) for t in times])
        frames = np.stack(frames)
        masks  = np.stack(masks)

        context        = frames[:self.T_in]
        target_abs     = frames[self.T_in:]
        last_ctx       = context[-1:]
        target_residual = target_abs - last_ctx

        lead_times = np.arange(1, self.T_out + 1, dtype=np.float32)

        if self.augment and np.random.rand() > 0.5:
            context         = context[:, :, :, ::-1].copy()
            target_residual = target_residual[:, :, :, ::-1].copy()
            last_ctx        = last_ctx[:, :, :, ::-1].copy()

        li_idx = self.channel_list.index("li") if "li" in self.channel_list else None
        if li_idx is not None:
            li_abs  = target_abs[:, li_idx]
            li_phys = li_abs * self._norm_std[li_idx] + self._norm_mean[li_idx]
            if self._cbrt_mask[li_idx]:
                li_phys = np.power(np.clip(li_phys, 0.0, None), 3)
            li_density = float((li_phys >= 5.0/255.0).mean())
        else:
            li_density = 0.0

        ctx = context[:, self.ctx_idx, :, :]

        _li_in_ctx = (li_idx is not None and li_idx in self.ctx_idx)
        if self.binary_li_ctx and _li_in_ctx:
            _li_ctx_pos = self.ctx_idx.index(li_idx)
            li_norm = ctx[:, _li_ctx_pos]
            li_phys = li_norm * self._norm_std[li_idx] + self._norm_mean[li_idx]
            if self._cbrt_mask[li_idx]:
                li_phys = np.power(np.clip(li_phys, 0.0, None), 3)
            li_bin = (li_phys >= 5.0 / 255.0).astype(np.float32)
            context_out = np.concatenate([ctx, li_bin[:, None, :, :]], axis=1)
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


def compute_li_sample_weights_packed(dataset, oversample_factor=5.0,
                                     density_percentile=75.0):
    """
    Packed-native equivalent of dataset.compute_li_sample_weights.

    The original directly accesses dataset.index[mid_t]["li"] and opens
    the JPEG file by path -- both are JPEG-pipeline-specific and do not
    exist on PackedMETSATDataset (no per-timestep file paths, just memmap
    rows). Reimplemented to read the SAME quantity (mean LI density of the
    middle target frame per sequence) directly from the memmap -- same
    percentile-threshold/oversample logic, just a different (faster, no
    decode) read path. Kept as a close mirror of the original for review.
    """
    ch_list = getattr(dataset, "channel_list", None)
    if ch_list is None or "li" not in ch_list:
        logger.warning("compute_li_sample_weights_packed: no LI channel, uniform weights")
        return np.ones(len(dataset), dtype=np.float64)
    if len(dataset) == 0:
        return np.zeros(0, dtype=np.float64)

    li_idx = ch_list.index("li")
    packed_li_idx = dataset._packed_ch_idx[li_idx]

    densities = np.zeros(len(dataset), dtype=np.float32)
    for i, seq_times in enumerate(dataset.valid_sequences):
        target_times = seq_times[dataset.T_in:]
        mid_t = target_times[len(target_times) // 2]
        row = dataset._row_of.get(mid_t)
        if row is None or dataset._mask[row, packed_li_idx] == 0:
            continue
        arr = dataset._frames[row, packed_li_idx]   # (H,W) uint8
        densities[i] = float((arr > 0).mean())

    threshold = float(np.percentile(densities, density_percentile))
    weights = np.where(densities >= threshold, float(oversample_factor), 1.0).astype(np.float64)
    n_high = int((densities >= threshold).sum())
    logger.info(f"  [packed] LI density threshold (p{density_percentile:.0f}): "
               f"{threshold:.4f}  high-density: {n_high}, low: {len(dataset)-n_high}")
    return weights


def make_dataloaders_packed(
    train_packed_dirs: List[str],   # one _packed dir per region
    channel_list:      List[str],
    T_in:               int = 6,
    T_out:              int = 36,
    dt_min:             int = 10,
    batch_size:         int = 4,
    num_workers:        int = 4,
    stats_roots:        Optional[List[str]] = None,   # original JPEG roots, for stats
    stat_path:          Optional[str] = None,
    max_samples:        Optional[int] = None,
    train_val_split:    float = 0.7,
    oversample_factor:  float = 5.0,
    density_percentile: float = 75.0,
    binary_li_ctx:      bool = True,
    ctx_channels:        Optional[List[str]] = None,
):
    """
    Packed-data equivalent of dataset.make_dataloaders. Same temporal
    per-region train/val split, same weighted-oversampling scheme -- only
    the underlying per-sample read path differs (memmap instead of JPEG).
    """
    import copy
    from torch.utils.data import DataLoader, WeightedRandomSampler

    assert 0.0 < train_val_split < 1.0
    stats_roots = stats_roots or [None] * len(train_packed_dirs)

    datasets = []
    for packed_dir, stats_root in zip(train_packed_dirs, stats_roots):
        ds = PackedMETSATDataset(
            packed_dir, channel_list=channel_list, T_in=T_in, T_out=T_out,
            dt_min=dt_min, stat_path=stat_path, stats_root=stats_root,
            augment=False, max_samples=max_samples,
            binary_li_ctx=binary_li_ctx, ctx_channels=ctx_channels,
        )
        datasets.append(ds)
    shared_stats = datasets[0].stats

    train_parts, val_parts = [], []
    for ds in datasets:
        all_seqs = ds.valid_sequences
        n = len(all_seqs)
        n_train = max(1, int(n * train_val_split))
        train_seqs, val_seqs = all_seqs[:n_train], all_seqs[n_train:]

        train_ds = ds
        train_ds.valid_sequences = train_seqs
        train_ds.augment = True
        train_parts.append(train_ds)

        val_ds = copy.copy(ds)
        val_ds.valid_sequences = val_seqs
        val_ds.augment = False
        val_parts.append(val_ds)

        logger.info(f"  {ds.root}: {n} sequences -> {len(train_seqs)} train / "
                   f"{len(val_seqs)} val (split={train_val_split:.0%})")
        if len(train_seqs) == 0:
            raise RuntimeError(f"Packed region '{ds.root}' has ZERO training "
                              f"sequences (n={n}).")

    class _ConcatDS(Dataset):
        def __init__(self, dsets):
            self.datasets = dsets
            self.lengths = [len(d) for d in dsets]
            self.cumlen = np.cumsum([0] + self.lengths)
        def __len__(self): return int(self.cumlen[-1])
        def __getitem__(self, idx):
            ds_idx = int(np.searchsorted(self.cumlen[1:], idx, side="right"))
            return self.datasets[ds_idx][idx - self.cumlen[ds_idx]]

    train_combined = _ConcatDS(train_parts)
    val_combined = _ConcatDS(val_parts)
    logger.info(f"Total (packed) -- train: {len(train_combined)}, val: {len(val_combined)}")

    all_weights = np.concatenate([
        compute_li_sample_weights_packed(ds, oversample_factor=oversample_factor,
                                         density_percentile=density_percentile)
        for ds in train_parts
    ])
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(all_weights).double(),
        num_samples=len(train_combined), replacement=True,
    )
    train_loader = DataLoader(train_combined, batch_size=batch_size,
                              sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_combined, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, shared_stats


def make_test_loader_packed(
    test_packed_dirs: List[str],
    channel_list:      List[str],
    T_in:               int = 6,
    T_out:              int = 36,
    dt_min:             int = 10,
    batch_size:         int = 4,
    num_workers:        int = 4,
    stats:              Optional[Dict] = None,
    stats_roots:        Optional[List[str]] = None,
    stat_path:          Optional[str] = None,
    binary_li_ctx:      bool = True,
    ctx_channels:        Optional[List[str]] = None,
):
    from torch.utils.data import DataLoader
    stats_roots = stats_roots or [None] * len(test_packed_dirs)
    datasets = []
    for packed_dir, stats_root in zip(test_packed_dirs, stats_roots):
        ds = PackedMETSATDataset(
            packed_dir, channel_list=channel_list, T_in=T_in, T_out=T_out,
            dt_min=dt_min, stats=stats, stat_path=stat_path, stats_root=stats_root,
            augment=False, binary_li_ctx=binary_li_ctx, ctx_channels=ctx_channels,
        )
        datasets.append(ds)

    class _ConcatDS(Dataset):
        def __init__(self, dsets):
            self.datasets = dsets
            self.lengths = [len(d) for d in dsets]
            self.cumlen = np.cumsum([0] + self.lengths)
        def __len__(self): return int(self.cumlen[-1])
        def __getitem__(self, idx):
            ds_idx = int(np.searchsorted(self.cumlen[1:], idx, side="right"))
            return self.datasets[ds_idx][idx - self.cumlen[ds_idx]]

    combined = _ConcatDS(datasets)
    if len(combined) == 0:
        raise RuntimeError("No valid sequences found in any packed test region")
    return DataLoader(combined, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
