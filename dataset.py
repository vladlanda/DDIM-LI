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
import re
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

logger = logging.getLogger(__name__)

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
    """Return per-channel mean/std (or median/iqr for LI)."""
    if stat_path and os.path.exists(stat_path):
        with open(stat_path) as f:
            return json.load(f)

    logger.info("Computing dataset statistics …")
    accum = defaultdict(list)
    all_files = list(Path(root).rglob("*.jpg"))
    np.random.shuffle(all_files)
    for fp in all_files[:n_samples]:
        ch = fp.stem.split("_")[-1]
        if ch not in channel_list:
            continue
        img = np.array(Image.open(fp).convert("L"), dtype=np.float32) / 255.0
        accum[ch].append(img.ravel())

    stats = {}
    for ch, arrays in accum.items():
        flat = np.concatenate(arrays)
        if ch == "li":
            # LI is very sparse – use cube-root transform stats
            flat = np.cbrt(flat)
            stats[ch] = {"mean": float(flat.mean()), "std": float(flat.std() + 1e-6),
                         "transform": "cbrt"}
        else:
            stats[ch] = {"mean": float(flat.mean()), "std": float(flat.std() + 1e-6),
                         "transform": "linear"}

    if stat_path:
        with open(stat_path, "w") as f:
            json.dump(stats, f, indent=2)
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
# Timestamp parsing
# -------------------------------------------------------------------
TS_FMT = "%Y%m%dT%H%M%SZ"

def parse_filename(path: Path) -> Optional[Tuple[str, datetime, datetime, str]]:
    """Parse {id}_{start}_{end}_{channel}.jpg -> (id, start_dt, end_dt, channel)"""
    name = path.stem
    parts = name.rsplit("_", 3)
    if len(parts) != 4:
        return None
    fid, start_s, end_s, ch = parts
    try:
        start_dt = datetime.strptime(start_s, TS_FMT)
        end_dt   = datetime.strptime(end_s,   TS_FMT)
    except ValueError:
        return None
    return fid, start_dt, end_dt, ch


# -------------------------------------------------------------------
# Index builder
# -------------------------------------------------------------------
def build_index(root: str) -> Dict[datetime, Dict[str, Path]]:
    """
    Returns: { start_dt: { channel: Path, ... }, ... }
    Only timesteps that have BOTH ir AND li are included.
    """
    raw = defaultdict(dict)
    for fp in Path(root).rglob("*.jpg"):
        parsed = parse_filename(fp)
        if parsed is None:
            continue
        _, start_dt, _, ch = parsed
        raw[start_dt][ch] = fp

    index = {dt: chs for dt, chs in raw.items()
             if all(c in chs for c in REQUIRED_CHANNELS)}
    return index


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
        T_in: int = 6,           # context frames
        T_out: int = 36,         # 36 × 10 min = 6 hours
        dt_min: int = 10,        # minutes between frames
        img_size: Tuple[int,int] = (256, 256),
        stats: Optional[Dict] = None,
        stat_path: Optional[str] = None,
        augment: bool = True,
        max_samples: Optional[int] = None,   # limit sequences for fast testing
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

        # Build valid sequence start indices
        self.valid_starts = self._build_valid_starts()

        if max_samples is not None and max_samples < len(self.valid_starts):
            # Evenly spaced subset so we sample across the full time range,
            # not just the first N (which could be a single storm event)
            step = len(self.valid_starts) // max_samples
            self.valid_starts = self.valid_starts[::step][:max_samples]
            logger.info(f"Dataset '{root}': {len(self.valid_starts)} sequences "
                        f"(capped at max_samples={max_samples})")
        else:
            logger.info(f"Dataset '{root}': {len(self.valid_starts)} valid sequences")

    def _build_valid_starts(self) -> List[datetime]:
        seq_len = self.T_in + self.T_out
        valid = []
        time_set = set(self.sorted_times)
        for t in self.sorted_times:
            seq = [t + i * self.dt for i in range(seq_len)]
            if all(s in time_set for s in seq):
                valid.append(t)
        return valid

    def __len__(self):
        return len(self.valid_starts)

    def _load_frame(self, dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (C, H, W) normalised frame and (C,) mask."""
        chs = self.index[dt]
        h, w = self.img_size
        frame = np.zeros((self.C, h, w), dtype=np.float32)
        mask  = np.zeros(self.C, dtype=np.float32)

        for i, ch in enumerate(self.channel_list):
            if ch in chs:
                img = Image.open(chs[ch]).convert("L").resize((w, h), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                if ch in self.stats:
                    arr = normalize(arr, self.stats, ch)
                else:
                    arr = (arr - 0.5) / 0.5
                frame[i] = arr
                mask[i]  = 1.0
            else:
                frame[i] = CHANNEL_FILL_VALUES.get(ch, 0.0)
                mask[i]  = 0.0

        return frame, mask

    def __getitem__(self, idx):
        t0 = self.valid_starts[idx]
        times = [t0 + i * self.dt for i in range(self.T_in + self.T_out)]

        frames, masks = zip(*[self._load_frame(t) for t in times])
        frames = np.stack(frames)  # (T_in+T_out, C, H, W)
        masks  = np.stack(masks)   # (T_in+T_out, C)

        context = frames[:self.T_in]                     # (T_in, C, H, W)
        target_abs = frames[self.T_in:]                  # (T_out, C, H, W)

        # Residual: target - last context frame
        last_ctx = context[-1:]                          # (1, C, H, W)
        target_residual = target_abs - last_ctx          # (T_out, C, H, W)

        lead_times = np.arange(1, self.T_out + 1, dtype=np.float32)  # [1..T_out] × dt_min

        # Random horizontal flip augmentation
        if self.augment and np.random.rand() > 0.5:
            context         = context[:, :, :, ::-1].copy()
            target_residual = target_residual[:, :, :, ::-1].copy()

        return {
            "context":    torch.from_numpy(context),
            "target":     torch.from_numpy(target_residual),
            "lead_times": torch.from_numpy(lead_times),
            "ctx_mask":   torch.from_numpy(masks[:self.T_in]),
            "tgt_mask":   torch.from_numpy(masks[self.T_in:]),
            "last_ctx":   torch.from_numpy(last_ctx[0]),   # (C,H,W) for denorm at inference
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

def make_dataloaders(
    train_roots:     List[str],
    channel_list:    List[str],
    T_in:            int   = 6,
    T_out:           int   = 36,
    img_size:        Tuple[int, int] = (256, 256),
    batch_size:      int   = 4,
    num_workers:     int   = 4,
    stat_path:       Optional[str] = None,
    max_samples:     Optional[int] = None,
    train_val_split: float = 0.7,
):
    """
    Builds train and validation loaders from train_roots only.

    The split is done TEMPORALLY per region — the first `train_val_split`
    fraction of each region's sequences go to train, the rest to val.
    This is the correct approach for time-series data: you must never
    let future frames leak into the training set via random shuffling.

    val_roots (your held-out test regions) are NOT touched here.
    Use make_test_loader() after training for final evaluation on those.
    """
    assert 0.0 < train_val_split < 1.0, "train_val_split must be in (0, 1)"

    # Build one dataset per region to get the full valid_starts list,
    # then slice temporally before constructing the final split datasets.
    train_starts_per_root: List[List] = []
    val_starts_per_root:   List[List] = []

    # First pass: build index + stats using all train_roots
    full_ds = MultiRegionDataset(
        train_roots, channel_list=channel_list,
        T_in=T_in, T_out=T_out, img_size=img_size,
        stat_path=stat_path, augment=False,
        max_samples=max_samples,
    )
    shared_stats = full_ds.datasets[0].stats

    for ds in full_ds.datasets:
        n       = len(ds.valid_starts)
        n_train = max(1, int(n * train_val_split))
        # Temporal split: first n_train → train, remainder → val
        train_starts_per_root.append(ds.valid_starts[:n_train])
        val_starts_per_root.append(ds.valid_starts[n_train:])
        logger.info(
            f"  {ds.root}: {n} sequences → "
            f"{n_train} train / {n - n_train} val  "
            f"(split={train_val_split:.0%})"
        )

    # Second pass: build split datasets by injecting pre-sliced valid_starts
    train_parts, val_parts = [], []
    for i, root in enumerate(train_roots):
        for starts, augment, parts in [
            (train_starts_per_root[i], True,  train_parts),
            (val_starts_per_root[i],   False, val_parts),
        ]:
            if not starts:
                continue
            d = METSATDataset(
                root, channel_list=channel_list,
                T_in=T_in, T_out=T_out, img_size=img_size,
                stats=shared_stats, augment=augment,
            )
            d.valid_starts = starts   # inject the pre-sliced list
            parts.append(d)

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

    train_ds = _ConcatDS(train_parts)
    val_ds   = _ConcatDS(val_parts)

    logger.info(f"Total — train: {len(train_ds)} sequences, val: {len(val_ds)} sequences")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
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
    Loader for the held-out test regions (val_roots in the config).
    Only call this after training is complete.
    Stats must be passed in from the training set — never refit on test data.
    """
    test_ds = MultiRegionDataset(
        test_roots, channel_list=channel_list,
        T_in=T_in, T_out=T_out, img_size=img_size,
        stats=stats, augment=False,
        max_samples=max_samples,
    )
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
