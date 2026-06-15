#!/usr/bin/env python3
"""
homogeneous_operator_boosting.py

Full-size neural-operator baselines versus boosted ensembles of tiny same-family
neural operators across FNO, DeepONet, and CNO.

Comparison performed for each dataset D and model family f:

  Full baseline:
      G_full^f(a)

  Tiny boosted ensemble:
      G_boost^f(a) = G_init + sum_{m=1}^M eta_m H_m^f(a)

where H_m^f is a tiny model from the same family trained stagewise on the current
normalized residual. Because y is normalized using the training split, G_init=0
is the empirical mean predictor in normalized coordinates.

Positive improvement means the boosted tiny ensemble beats the full-size
standalone baseline:

  improvement_pct = 100 * (full_rel_l2 - boosted_rel_l2) / full_rel_l2

Data sources:
  1. Existing random_operator_toys standardized datasets:
       <common_data_root>/common/*.h5 with datasets `a` and `u`.
  2. Additional benchmark trajectory data using the same layout/globs as the
     diagnostic-suite script:
       APEBench NS2D, PDEBench shallow water 2D, The Well active matter 2D,
       PDEBench 3D compressible NS, The Well 3D MHD.

The model implementations below are intentionally substantial:
  - FNO: multidimensional spectral convolution with all low-frequency corners
    for rFFT geometry, coordinate conditioning, residual spectral blocks.
  - DeepONet: convolutional branch encoder, Fourier-feature coordinate trunk,
    learned basis coefficients, full-grid evaluation in 1D/2D/3D.
  - CNO: multiresolution convolutional neural operator with coordinate lifting,
    residual blocks, anti-aliased downsampling, skip-connected decoding, and
    interpolation-based continuous-resolution behavior.

This is still a research driver, not a hyperparameter-tuned final benchmark.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

EPS = 1.0e-8

COMMON_DATASETS = [
    "1d_advection",
    "1d_burgers",
    "1d_reacdiff",
    "pdebench_darcy",
    "pdebench_2d_reacdiff",
]

EXTERNAL_DATASETS = ["ns2d", "shallow2d", "active2d", "cns3d", "mhd3d"]
DEFAULT_DATASETS = COMMON_DATASETS + EXTERNAL_DATASETS
DEFAULT_FAMILIES = ["fno", "deeponet", "cno"]


# =============================================================================
# Reproducibility / metrics
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    b = pred.shape[0]
    return torch.linalg.vector_norm((pred.reshape(b, -1) - target.reshape(b, -1)), dim=1) / (
        torch.linalg.vector_norm(target.reshape(b, -1), dim=1) + eps
    )


def pct_improvement(baseline: float, method: float) -> float:
    if not np.isfinite(baseline) or baseline <= 0:
        return float("nan")
    return 100.0 * (baseline - method) / baseline


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# =============================================================================
# Dataset specs and generic benchmark readers
# =============================================================================

@dataclass
class TaskSpec:
    name: str
    label: str
    dim: int
    source: str  # common or external
    globs: Dict[str, List[str]] = field(default_factory=dict)
    max_channels: int = 8
    append_dt_channel: bool = False
    internal_split: set[str] = field(default_factory=set)


def build_task_specs() -> Dict[str, TaskSpec]:
    specs: Dict[str, TaskSpec] = {}
    for name in COMMON_DATASETS:
        dim = 1 if name.startswith("1d_") else 2
        specs[name] = TaskSpec(
            name=name,
            label=f"standardized {name}",
            dim=dim,
            source="common",
            max_channels=16,
            append_dt_channel=False,
        )

    specs.update(
        {
            "ns2d": TaskSpec(
                name="ns2d",
                label="2D incompressible NS / APEBench",
                dim=2,
                source="external",
                globs={
                    "train": ["apebench/apebench_incompressible_ns_2d/data/*train.npy"],
                    "val": ["apebench/apebench_incompressible_ns_2d/data/*train.npy"],
                    "test": ["apebench/apebench_incompressible_ns_2d/data/*test.npy"],
                },
                max_channels=4,
                append_dt_channel=True,
                internal_split={"train", "val"},
            ),
            "shallow2d": TaskSpec(
                name="shallow2d",
                label="2D shallow water / PDEBench",
                dim=2,
                source="external",
                globs={
                    "train": ["pdebench/pdebench_shallow_water_2d/*.h5"],
                    "val": ["pdebench/pdebench_shallow_water_2d/*.h5"],
                    "test": ["pdebench/pdebench_shallow_water_2d/*.h5"],
                },
                max_channels=4,
                append_dt_channel=True,
                internal_split={"train", "val", "test"},
            ),
            "active2d": TaskSpec(
                name="active2d",
                label="2D active matter / The Well",
                dim=2,
                source="external",
                globs={
                    "train": ["the_well/datasets/active_matter/data/train/*.hdf5"],
                    "val": ["the_well/datasets/active_matter/data/valid/*.hdf5"],
                    "test": ["the_well/datasets/active_matter/data/test/*.hdf5"],
                },
                max_channels=8,
                append_dt_channel=True,
                internal_split=set(),
            ),
            "cns3d": TaskSpec(
                name="cns3d",
                label="3D compressible NS / PDEBench",
                dim=3,
                source="external",
                globs={
                    "train": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
                    "val": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
                    "test": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
                },
                max_channels=5,
                append_dt_channel=True,
                internal_split={"train", "val", "test"},
            ),
            "mhd3d": TaskSpec(
                name="mhd3d",
                label="3D MHD / The Well",
                dim=3,
                source="external",
                globs={
                    "train": ["the_well/datasets/MHD_64/data/train/*.hdf5"],
                    "val": ["the_well/datasets/MHD_64/data/valid/*.hdf5"],
                    "test": ["the_well/datasets/MHD_64/data/test/*.hdf5"],
                },
                max_channels=8,
                append_dt_channel=True,
                internal_split=set(),
            ),
        }
    )
    return specs


def _numeric_h5_datasets(path: Path) -> List[Tuple[str, Tuple[int, ...], str]]:
    out: List[Tuple[str, Tuple[int, ...], str]] = []
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                try:
                    if len(obj.shape) >= 3 and np.issubdtype(obj.dtype, np.number):
                        out.append((name, tuple(int(x) for x in obj.shape), str(obj.dtype)))
                except Exception:
                    pass
        f.visititems(visit)
    return out


def _choose_h5_dataset(path: Path, dim: int) -> str:
    candidates = _numeric_h5_datasets(path)
    if not candidates:
        raise RuntimeError(f"No numeric HDF5 trajectory datasets found in {path}")
    ranked = []
    for name, shape, _dtype in candidates:
        lname = name.lower()
        score = 0
        if len(shape) >= dim + 2:
            score += 50
        if len(shape) >= dim + 3:
            score += 50
        if len(shape) >= 2 and shape[0] > 1:
            score += 10
        if len(shape) >= 2 and shape[1] > 1:
            score += 10
        if any(s in lname for s in ("field", "data", "solution", "tensor", "trajectory", "states", "u")):
            score += 25
        if any(s in lname for s in ("coord", "grid", "time", "x-coordinate", "y-coordinate", "param", "mask")):
            score -= 100
        if shape[-1] <= 64:
            score += 5
        score += min(50, int(math.log10(max(1, int(np.prod(shape))))))
        ranked.append((score, name, shape))
    ranked.sort(reverse=True)
    return ranked[0][1]


def _nested_h5_trajectory_keys(path: Path, dim: int) -> List[Tuple[str, Tuple[int, ...]]]:
    out: List[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            try:
                if not np.issubdtype(obj.dtype, np.number):
                    return
                shape = tuple(int(x) for x in obj.shape)
                lname = name.lower()
                if not (lname.endswith("/data") or lname.endswith("/tensor") or lname.endswith("/solution")):
                    return
                if len(shape) in {dim + 1, dim + 2} and shape[0] > 1 and max(shape[1:]) > 8:
                    out.append((name, shape))
            except Exception:
                pass
        f.visititems(visit)
    out.sort(key=lambda x: x[0])
    return out


def _top_level_cfd_keys(path: Path, dim: int) -> Tuple[List[str], Optional[Tuple[int, ...]]]:
    preferred = [
        "density", "pressure", "Vx", "Vy", "Vz", "vx", "vy", "vz",
        "velocity_x", "velocity_y", "velocity_z", "x-velocity", "y-velocity", "z-velocity",
    ]
    with h5py.File(path, "r") as f:
        shapes: Dict[str, Tuple[int, ...]] = {}
        for k in preferred:
            if k in f and isinstance(f[k], h5py.Dataset):
                try:
                    if np.issubdtype(f[k].dtype, np.number):
                        shape = tuple(int(x) for x in f[k].shape)
                        if len(shape) == dim + 2 and shape[0] > 1 and shape[1] > 1:
                            shapes[k] = shape
                except Exception:
                    pass
        if not shapes:
            return [], None
        counts: Dict[Tuple[int, ...], int] = {}
        for s in shapes.values():
            counts[s] = counts.get(s, 0) + 1
        shape = max(counts.items(), key=lambda kv: kv[1])[0]
        keys = [k for k in preferred if shapes.get(k) == shape]
        return keys, shape


def _to_channels_first_state(arr: np.ndarray, dim: int, max_channels: int) -> torch.Tensor:
    x = np.asarray(arr, dtype=np.float32)
    while x.ndim > dim + 1 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == dim:
        x = x[..., None]
    if x.ndim > dim + 1:
        # Prefer spatial-first, channel-last layout when ambiguous.
        if x.shape[-1] <= max_channels or x.shape[-1] <= x.shape[0]:
            spatial = x.shape[:dim]
            x = x.reshape(*spatial, -1)
        elif x.shape[0] <= max_channels:
            x = x.reshape(x.shape[0], *x.shape[-dim:])
        else:
            spatial = x.shape[:dim]
            x = x.reshape(*spatial, -1)
    if x.shape[-1] <= max_channels:
        x = np.moveaxis(x, -1, 0)
    elif x.shape[0] <= max_channels:
        pass
    else:
        x = np.moveaxis(x, -1, 0)
    if x.shape[0] > max_channels:
        x = x[:max_channels]
    return torch.from_numpy(np.ascontiguousarray(x))


def _spatial_limit_cf(x: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0:
        return x.contiguous()
    slices: List[slice] = [slice(None)]
    for n in x.shape[1:]:
        if n <= limit:
            slices.append(slice(None))
        else:
            step = int(math.ceil(n / limit))
            slices.append(slice(0, n, step))
    y = x[tuple(slices)]
    slices = [slice(None)]
    for n in y.shape[1:]:
        if n <= limit:
            slices.append(slice(None))
        else:
            start = (n - limit) // 2
            slices.append(slice(start, start + limit))
    return y[tuple(slices)].contiguous()


class BaseReader:
    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int, split_within_file: bool):
        self.path = Path(path)
        self.task = task
        self.split = "val" if split == "valid" else split
        self.spatial_limit = spatial_limit
        self.split_within_file = split_within_file
        self.n_traj = 0
        self.n_time = 0

    def state(self, traj: int, t: int) -> torch.Tensor:
        raise NotImplementedError

    def traj_indices_for_split(self) -> List[int]:
        if self.n_traj <= 0:
            return []
        if not self.split_within_file:
            return list(range(self.n_traj))
        n_train = max(1, int(0.80 * self.n_traj))
        n_val = max(1, int(0.10 * self.n_traj))
        if self.split == "train":
            idx = list(range(0, n_train))
        elif self.split in {"val", "valid", "validation"}:
            idx = list(range(n_train, min(self.n_traj, n_train + n_val)))
        else:
            idx = list(range(min(self.n_traj, n_train + n_val), self.n_traj))
        return idx if idx else list(range(self.n_traj))


class NpyReader(BaseReader):
    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int, split_within_file: bool):
        super().__init__(path, task, split, spatial_limit, split_within_file)
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            keys = sorted([k for k in arr.files if np.asarray(arr[k]).ndim >= task.dim + 2])
            if not keys:
                raise RuntimeError(f"No trajectory array found in {path}")
            arr = np.asarray(arr[keys[0]])
        else:
            arr = np.asarray(arr)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == task.dim + 2:
            # Single trajectory: (T,*spatial,C?)
            arr = arr[None]
        self.arr = arr
        self.n_traj = int(arr.shape[0])
        self.n_time = int(arr.shape[1])

    def state(self, traj: int, t: int) -> torch.Tensor:
        x = _to_channels_first_state(self.arr[traj, t], self.task.dim, self.task.max_channels)
        return _spatial_limit_cf(x, self.spatial_limit)


class H5Reader(BaseReader):
    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int, split_within_file: bool):
        super().__init__(path, task, split, spatial_limit, split_within_file)
        self._h5: Optional[h5py.File] = None
        self.mode = "batched"
        self.dataset_key: Optional[str] = None
        self.dataset_keys: List[str] = []
        cfd_keys, cfd_shape = _top_level_cfd_keys(self.path, task.dim)
        nested = _nested_h5_trajectory_keys(self.path, task.dim)
        if cfd_keys and cfd_shape is not None:
            self.mode = "cfd"
            self.dataset_keys = cfd_keys[: task.max_channels]
            self.shape = cfd_shape
            self.n_traj = int(cfd_shape[0])
            self.n_time = int(cfd_shape[1])
        elif nested:
            self.mode = "nested"
            self.dataset_keys = [k for k, _ in nested]
            self.shape = nested[0][1]
            self.n_traj = len(self.dataset_keys)
            self.n_time = int(self.shape[0])
        else:
            self.mode = "batched"
            self.dataset_key = _choose_h5_dataset(self.path, task.dim)
            with h5py.File(self.path, "r") as f:
                self.shape = tuple(int(x) for x in f[self.dataset_key].shape)
            if len(self.shape) == task.dim + 1:
                self.n_traj = 1
                self.n_time = int(self.shape[0])
            else:
                self.n_traj = int(self.shape[0])
                self.n_time = int(self.shape[1]) if len(self.shape) >= task.dim + 2 else 0
        if self.n_time < 2:
            raise RuntimeError(f"HDF5 reader found too few time steps in {path}: mode={self.mode}, shape={getattr(self, 'shape', None)}")

    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def state(self, traj: int, t: int) -> torch.Tensor:
        f = self.h5()
        if self.mode == "cfd":
            chans = [np.asarray(f[k][traj, t], dtype=np.float32) for k in self.dataset_keys]
            x = torch.from_numpy(np.ascontiguousarray(np.stack(chans, axis=0)))
            return _spatial_limit_cf(x[: self.task.max_channels], self.spatial_limit)
        if self.mode == "nested":
            key = self.dataset_keys[traj % len(self.dataset_keys)]
            x = _to_channels_first_state(np.asarray(f[key][t], dtype=np.float32), self.task.dim, self.task.max_channels)
            return _spatial_limit_cf(x, self.spatial_limit)
        assert self.dataset_key is not None
        dset = f[self.dataset_key]
        if len(self.shape) == self.task.dim + 1:
            arr = np.asarray(dset[t], dtype=np.float32)
        else:
            arr = np.asarray(dset[traj, t], dtype=np.float32)
        x = _to_channels_first_state(arr, self.task.dim, self.task.max_channels)
        return _spatial_limit_cf(x, self.spatial_limit)


class TrajectoryCollection:
    def __init__(self, data_root: Path, task: TaskSpec, split: str, spatial_limit: int):
        self.data_root = Path(data_root)
        self.task = task
        self.split = "val" if split == "valid" else split
        self.spatial_limit = spatial_limit
        self.readers: List[BaseReader] = []
        for pat in task.globs[self.split]:
            for p in sorted(glob.glob(str(self.data_root / pat))):
                path = Path(p)
                # For files that must be internally split, force that behavior. Otherwise,
                # use all trajectories when the file path explicitly names the split.
                split_within = self.split in task.internal_split
                if not split_within:
                    low = path.as_posix().lower()
                    split_tokens = [f"/{self.split}/", f"_{self.split}", f"{self.split}.", f"{self.split}_"]
                    if self.split == "val":
                        split_tokens += ["/valid/", "_valid", "validation"]
                    force_all = any(tok in low for tok in split_tokens)
                    split_within = not force_all
                try:
                    if path.suffix.lower() in {".npy", ".npz"}:
                        self.readers.append(NpyReader(path, task, self.split, spatial_limit, split_within))
                    elif path.suffix.lower() in {".h5", ".hdf5", ".hdf"}:
                        self.readers.append(H5Reader(path, task, self.split, spatial_limit, split_within))
                except Exception as e:
                    print(f"[WARN] Could not open {path}: {e}")
        if not self.readers:
            raise RuntimeError(f"No readable files for task={task.name}, split={self.split}, patterns={task.globs[self.split]}")


def cf_to_cl(x: torch.Tensor) -> torch.Tensor:
    # [C, *spatial] -> [*spatial, C]
    return x.movedim(0, -1).contiguous()


def make_tau_channel_cf(x_cf: torch.Tensor, tau: float) -> torch.Tensor:
    spatial = tuple(int(s) for s in x_cf.shape[1:])
    return torch.full((1, *spatial), float(tau), dtype=x_cf.dtype)


def sample_trajectory_pairs(
    collection: TrajectoryCollection,
    *,
    max_pairs: int,
    max_delta: int,
    seed: int,
    append_dt_channel: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    items: List[Tuple[int, int, int, int]] = []
    for ridx, r in enumerate(collection.readers):
        for traj in r.traj_indices_for_split():
            for t0 in range(0, r.n_time - 1):
                for dt in range(1, min(max_delta, r.n_time - 1 - t0) + 1):
                    items.append((ridx, traj, t0, dt))
    if not items:
        raise RuntimeError(f"No trajectory pairs for {collection.task.name}/{collection.split}")
    rng = random.Random(seed)
    rng.shuffle(items)
    if max_pairs > 0:
        items = items[:max_pairs]

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for ridx, traj, t0, dt in items:
        r = collection.readers[ridx]
        x_cf = r.state(traj, t0).float()
        y_cf = r.state(traj, t0 + dt).float()
        if append_dt_channel:
            tau = float(dt) / float(max(1, max_delta))
            x_cf = torch.cat([x_cf, make_tau_channel_cf(x_cf, tau)], dim=0)
        xs.append(cf_to_cl(x_cf))
        ys.append(cf_to_cl(y_cf))
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


@dataclass
class LoadedProblem:
    name: str
    label: str
    source: str
    dim: int
    group: str
    cin: int
    cout: int
    spatial_shape: Tuple[int, ...]
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    y_test_raw: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def denorm_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)


def _split_train_val_tensors(x: torch.Tensor, y: torch.Tensor, val_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    if n < 4:
        raise RuntimeError("Need at least 4 samples for train/val split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(4, int(round(val_frac * n)))
    n_val = min(n_val, max(1, n // 2))
    val_idx = torch.tensor(perm[:n_val], dtype=torch.long)
    tr_idx = torch.tensor(perm[n_val:], dtype=torch.long)
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def _normalize_splits(
    name: str,
    label: str,
    source: str,
    dim: int,
    x_train_raw: torch.Tensor,
    y_train_raw: torch.Tensor,
    x_val_raw: torch.Tensor,
    y_val_raw: torch.Tensor,
    x_test_raw: torch.Tensor,
    y_test_raw: torch.Tensor,
) -> LoadedProblem:
    reduce_x = tuple(range(x_train_raw.ndim - 1))
    reduce_y = tuple(range(y_train_raw.ndim - 1))
    x_mean = x_train_raw.mean(dim=reduce_x, keepdim=True)
    x_std = x_train_raw.std(dim=reduce_x, keepdim=True).clamp_min(1e-6)
    y_mean = y_train_raw.mean(dim=reduce_y, keepdim=True)
    y_std = y_train_raw.std(dim=reduce_y, keepdim=True).clamp_min(1e-6)

    x_train = (x_train_raw - x_mean) / x_std
    y_train = (y_train_raw - y_mean) / y_std
    x_val = (x_val_raw - x_mean) / x_std
    y_val = (y_val_raw - y_mean) / y_std
    x_test = (x_test_raw - x_mean) / x_std
    y_test = (y_test_raw - y_mean) / y_std

    group = f"{dim}d"
    return LoadedProblem(
        name=name,
        label=label,
        source=source,
        dim=dim,
        group=group,
        cin=int(x_train.shape[-1]),
        cout=int(y_train.shape[-1]),
        spatial_shape=tuple(int(s) for s in x_train.shape[1:-1]),
        x_train=x_train.float(),
        y_train=y_train.float(),
        x_val=x_val.float(),
        y_val=y_val.float(),
        x_test=x_test.float(),
        y_test=y_test.float(),
        y_test_raw=y_test_raw.float(),
        x_mean=x_mean.float(),
        x_std=x_std.float(),
        y_mean=y_mean.float(),
        y_std=y_std.float(),
    )


def load_common_problem(name: str, task: TaskSpec, args: argparse.Namespace, seed: int) -> LoadedProblem:
    path = args.common_data_root / "common" / f"{name}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Missing standardized dataset: {path}")
    with h5py.File(path, "r") as f:
        a = torch.tensor(np.asarray(f["a"], dtype=np.float32))
        u = torch.tensor(np.asarray(f["u"], dtype=np.float32))
    n = min(int(args.max_samples), int(a.shape[0])) if args.max_samples > 0 else int(a.shape[0])
    rng = np.random.default_rng(seed)
    idx = torch.tensor(rng.permutation(a.shape[0])[:n], dtype=torch.long)
    a = a[idx]
    u = u[idx]
    n_test = max(8, int(round(args.test_frac * n)))
    n_test = min(n_test, max(1, n // 2))
    x_train_val = a[: n - n_test]
    y_train_val = u[: n - n_test]
    x_test_raw = a[n - n_test :]
    y_test_raw = u[n - n_test :]
    x_train_raw, y_train_raw, x_val_raw, y_val_raw = _split_train_val_tensors(
        x_train_val, y_train_val, args.val_frac, seed + 17
    )
    return _normalize_splits(
        name, task.label, "common", task.dim,
        x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw,
    )


def spatial_limit_for_dim(dim: int, args: argparse.Namespace) -> int:
    if dim == 1:
        return int(args.spatial_limit_1d)
    if dim == 2:
        return int(args.spatial_limit_2d)
    if dim == 3:
        return int(args.spatial_limit_3d)
    raise ValueError(dim)


def load_external_problem(name: str, task: TaskSpec, args: argparse.Namespace, seed: int) -> LoadedProblem:
    lim = spatial_limit_for_dim(task.dim, args)
    train_coll = TrajectoryCollection(args.external_data_root, task, "train", lim)
    val_coll = TrajectoryCollection(args.external_data_root, task, "val", lim)
    test_coll = TrajectoryCollection(args.external_data_root, task, "test", lim)

    x_train_raw, y_train_raw = sample_trajectory_pairs(
        train_coll,
        max_pairs=args.max_train_pairs,
        max_delta=args.max_delta,
        seed=seed + 101,
        append_dt_channel=task.append_dt_channel,
    )
    x_val_raw, y_val_raw = sample_trajectory_pairs(
        val_coll,
        max_pairs=args.max_val_pairs,
        max_delta=args.max_delta,
        seed=seed + 102,
        append_dt_channel=task.append_dt_channel,
    )
    x_test_raw, y_test_raw = sample_trajectory_pairs(
        test_coll,
        max_pairs=args.max_test_pairs,
        max_delta=args.max_delta,
        seed=seed + 103,
        append_dt_channel=task.append_dt_channel,
    )
    return _normalize_splits(
        name, task.label, "external", task.dim,
        x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw,
    )


def load_problem(name: str, args: argparse.Namespace, seed: int) -> LoadedProblem:
    specs = build_task_specs()
    if name not in specs:
        raise KeyError(f"Unknown dataset {name}. Choices: {sorted(specs)}")
    task = specs[name]
    if task.source == "common":
        return load_common_problem(name, task, args, seed)
    return load_external_problem(name, task, args, seed)


# =============================================================================
# Tensor layout helpers for models
# =============================================================================

def channels_last_to_first(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 1:
        return x.permute(0, 2, 1).contiguous()
    if dim == 2:
        return x.permute(0, 3, 1, 2).contiguous()
    if dim == 3:
        return x.permute(0, 4, 1, 2, 3).contiguous()
    raise ValueError(dim)


def channels_first_to_last(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 1:
        return x.permute(0, 2, 1).contiguous()
    if dim == 2:
        return x.permute(0, 2, 3, 1).contiguous()
    if dim == 3:
        return x.permute(0, 2, 3, 4, 1).contiguous()
    raise ValueError(dim)


def conv_cls(dim: int):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]


def pool_cls(dim: int):
    return {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}[dim]


def adaptive_pool_cls(dim: int):
    return {1: nn.AdaptiveAvgPool1d, 2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[dim]


def interpolate_mode(dim: int) -> str:
    return {1: "linear", 2: "bilinear", 3: "trilinear"}[dim]


def group_norm(channels: int) -> nn.GroupNorm:
    for g in [16, 8, 4, 2, 1]:
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def coordinate_grid_cf(x_cf: torch.Tensor) -> torch.Tensor:
    spatial = tuple(int(s) for s in x_cf.shape[2:])
    dim = len(spatial)
    axes = [torch.linspace(0.0, 1.0, n, device=x_cf.device, dtype=x_cf.dtype) for n in spatial]
    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack(mesh, dim=0).unsqueeze(0)
    return grid.expand(x_cf.shape[0], -1, *spatial)


def add_coords_cf(x_cf: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_cf, coordinate_grid_cf(x_cf)], dim=1)


# =============================================================================
# Model family implementations
# =============================================================================

class PointwiseMLP(nn.Module):
    def __init__(self, dim: int, in_ch: int, hidden_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        Conv = conv_cls(dim)
        self.net = nn.Sequential(
            Conv(in_ch, hidden_ch, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            Conv(hidden_ch, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, dim: int, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        Conv = conv_cls(dim)
        pad = dilation * (kernel_size // 2)
        self.conv1 = Conv(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = group_norm(out_ch)
        self.conv2 = Conv(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = group_norm(out_ch)
        self.skip = Conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class SpectralConvND(nn.Module):
    """Real-valued multidimensional FNO spectral convolution with rFFT corners."""

    def __init__(self, dim: int, in_ch: int, out_ch: int, modes: int | Sequence[int]):
        super().__init__()
        self.dim = dim
        if isinstance(modes, int):
            self.modes = tuple([int(modes)] * dim)
        else:
            self.modes = tuple(int(m) for m in modes)
            if len(self.modes) != dim:
                raise ValueError(f"modes must have length {dim}")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.sign_corners = [()] if dim == 1 else list(np.ndindex(*([2] * (dim - 1))))
        scale = 1.0 / math.sqrt(max(1, in_ch * out_ch))
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_ch, out_ch, *self.modes, dtype=torch.cfloat))
            for _ in self.sign_corners
        ])

    def _mode_counts(self, xft_shape: Sequence[int]) -> Tuple[int, ...]:
        counts: List[int] = []
        for ax in range(self.dim - 1):
            n = int(xft_shape[ax])
            counts.append(min(self.modes[ax], max(1, n // 2)))
        counts.append(min(self.modes[-1], int(xft_shape[-1])))
        return tuple(counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        spatial = tuple(int(n) for n in x.shape[2:])
        fft_dims = tuple(range(2, x.ndim))
        xft = torch.fft.rfftn(x, dim=fft_dims, norm="ortho")
        out_ft = torch.zeros((b, self.out_ch, *xft.shape[2:]), device=x.device, dtype=torch.cfloat)
        counts = self._mode_counts(xft.shape[2:])
        if any(m <= 0 for m in counts):
            return torch.fft.irfftn(out_ft, s=spatial, dim=fft_dims, norm="ortho")
        for corner_id, signs in enumerate(self.sign_corners):
            data_slices: List[slice] = [slice(None), slice(None)]
            weight_slices: List[slice] = [slice(None), slice(None)]
            for ax, sign in enumerate(signs):
                m = counts[ax]
                data_slices.append(slice(0, m) if sign == 0 else slice(-m, None))
                weight_slices.append(slice(0, m))
            m_last = counts[-1]
            data_slices.append(slice(0, m_last))
            weight_slices.append(slice(0, m_last))
            out_ft[tuple(data_slices)] = torch.einsum(
                "bi...,io...->bo...",
                xft[tuple(data_slices)],
                self.weights[corner_id][tuple(weight_slices)],
            )
        return torch.fft.irfftn(out_ft, s=spatial, dim=fft_dims, norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, dim: int, width: int, modes: int, dropout: float = 0.0):
        super().__init__()
        Conv = conv_cls(dim)
        self.spectral = SpectralConvND(dim, width, width, modes)
        self.pointwise = Conv(width, width, 1)
        self.norm = group_norm(width)
        self.mlp = PointwiseMLP(dim, width, 2 * width, width, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spectral(x) + self.pointwise(x)
        h = F.gelu(self.norm(h))
        h = self.mlp(h)
        return F.gelu(x + h)


class FNOOperator(nn.Module):
    def __init__(self, dim: int, in_channels: int, out_channels: int, width: int, modes: int, layers: int, dropout: float = 0.0):
        super().__init__()
        Conv = conv_cls(dim)
        self.dim = dim
        self.lift = Conv(in_channels + dim, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(dim, width, modes, dropout=dropout) for _ in range(layers)])
        self.proj = PointwiseMLP(dim, width, 2 * width, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xcf = channels_last_to_first(x, self.dim)
        h = self.lift(add_coords_cf(xcf))
        for block in self.blocks:
            h = block(h)
        return channels_first_to_last(self.proj(h), self.dim)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_frequencies: int = 8, include_input: bool = True):
        super().__init__()
        self.include_input = include_input
        self.register_buffer("freqs", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32), persistent=False)
        self.out_dim = (in_dim if include_input else 0) + 2 * in_dim * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=x.device, dtype=x.dtype)
        angles = 2.0 * math.pi * x.unsqueeze(-1) * freqs
        feats = [torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)]
        if self.include_input:
            feats.insert(0, x)
        return torch.cat(feats, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, layers: int, dropout: float = 0.0):
        super().__init__()
        if layers < 2:
            raise ValueError("MLP requires at least 2 layers")
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchEncoder(nn.Module):
    def __init__(self, dim: int, in_channels: int, width: int, levels: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        layers: List[nn.Module] = []
        c = in_channels + dim
        w = width
        for level in range(levels):
            layers.append(ResidualConvBlock(dim, c, w, dropout=dropout))
            layers.append(ResidualConvBlock(dim, w, w, dropout=dropout))
            c = w
            if level != levels - 1:
                Pool = pool_cls(dim)
                layers.append(Pool(kernel_size=2, stride=2))
                w = min(4 * width, 2 * w)
        self.net = nn.Sequential(*layers)
        self.pool = adaptive_pool_cls(dim)(1)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xcf = channels_last_to_first(x, self.dim)
        h = self.net(add_coords_cf(xcf))
        return self.pool(h).flatten(1)


class DeepONetOperator(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        width: int,
        latent: int,
        spatial_shape: Tuple[int, ...],
        branch_levels: int = 4,
        trunk_layers: int = 4,
        fourier_frequencies: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.spatial_shape = tuple(int(s) for s in spatial_shape)
        self.latent = int(latent)
        self.branch = BranchEncoder(dim, in_channels, width, branch_levels, dropout=dropout)
        self.branch_head = MLP(self.branch.out_channels, out_channels * self.latent, hidden_dim=max(width * 2, latent), layers=3, dropout=dropout)
        self.coord_features = FourierFeatures(dim, num_frequencies=fourier_frequencies, include_input=True)
        self.trunk = MLP(self.coord_features.out_dim, self.latent, hidden_dim=max(width * 2, latent), layers=trunk_layers, dropout=dropout)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _coords(self, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        axes = [torch.linspace(0.0, 1.0, n, device=device, dtype=dtype) for n in self.spatial_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack(mesh, dim=-1).reshape(-1, self.dim)
        return coords.view(1, coords.shape[0], self.dim).expand(b, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        coeff = self.branch_head(self.branch(x)).view(b, self.out_channels, self.latent)
        coords = self._coords(b, x.device, x.dtype)
        basis = self.trunk(self.coord_features(coords))
        y = torch.einsum("bcl,bnl->bnc", coeff, basis) / math.sqrt(float(self.latent))
        y = y + self.bias.view(1, 1, self.out_channels)
        return y.reshape(b, *self.spatial_shape, self.out_channels).contiguous()


class AntiAliasDownsample(nn.Module):
    def __init__(self, dim: int, channels: int):
        super().__init__()
        Conv = conv_cls(dim)
        self.dim = dim
        self.blur = Conv(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.mix = Conv(channels, channels, 1)
        with torch.no_grad():
            self.blur.weight.fill_(1.0 / (3 ** dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blur(x)
        if self.dim == 1:
            x = F.avg_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)
        elif self.dim == 2:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        elif self.dim == 3:
            x = F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)
        else:
            raise ValueError(self.dim)
        return self.mix(x)


class CNOResidualBlock(nn.Module):
    def __init__(self, dim: int, channels: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        Conv = conv_cls(dim)
        hidden = expansion * channels
        self.net = nn.Sequential(
            Conv(channels, hidden, 3, padding=1),
            group_norm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            Conv(hidden, channels, 3, padding=1),
            group_norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class CNOStage(nn.Module):
    def __init__(self, dim: int, channels: int, blocks: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[CNOResidualBlock(dim, channels, dropout=dropout) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CNOOperator(nn.Module):
    """Multiresolution convolutional neural operator with coordinate lifting."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        width: int,
        levels: int,
        blocks_per_level: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        Conv = conv_cls(dim)
        self.dim = dim
        self.levels = max(2, int(levels))
        channels = [min(width * (2 ** i), 4 * width) for i in range(self.levels)]
        self.lift = nn.Sequential(
            Conv(in_channels + dim, channels[0], 3, padding=1),
            group_norm(channels[0]),
            nn.GELU(),
            Conv(channels[0], channels[0], 1),
        )
        self.enc_stages = nn.ModuleList([CNOStage(dim, channels[i], blocks_per_level, dropout=dropout) for i in range(self.levels)])
        self.downs = nn.ModuleList([
            nn.Sequential(AntiAliasDownsample(dim, channels[i]), Conv(channels[i], channels[i + 1], 1))
            for i in range(self.levels - 1)
        ])
        self.bottleneck = CNOStage(dim, channels[-1], blocks_per_level + 1, dropout=dropout)
        self.up_mix = nn.ModuleList([
            Conv(channels[i + 1], channels[i], 1) for i in range(self.levels - 2, -1, -1)
        ])
        self.dec_stages = nn.ModuleList([
            CNOStage(dim, channels[i], blocks_per_level, dropout=dropout) for i in range(self.levels - 2, -1, -1)
        ])
        self.project = nn.Sequential(
            Conv(channels[0], channels[0], 3, padding=1),
            nn.GELU(),
            Conv(channels[0], out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xcf = channels_last_to_first(x, self.dim)
        h = self.lift(add_coords_cf(xcf))
        skips: List[torch.Tensor] = []
        for i, stage in enumerate(self.enc_stages):
            h = stage(h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        h = self.bottleneck(h)
        for j, (mix, dec) in enumerate(zip(self.up_mix, self.dec_stages)):
            skip = skips[-(j + 2)]
            h = F.interpolate(h, size=skip.shape[2:], mode=interpolate_mode(self.dim), align_corners=False)
            h = mix(h)
            h = dec(h + skip)
        return channels_first_to_last(self.project(h), self.dim)


# =============================================================================
# Model factory
# =============================================================================

def full_tiny_hparams(dim: int, args: argparse.Namespace, tiny: bool) -> Dict[str, int]:
    if tiny:
        width = args.tiny_width_3d if dim == 3 else args.tiny_width
        layers = args.tiny_layers
        modes = args.tiny_modes_3d if dim == 3 else args.tiny_modes
        latent = args.tiny_deeponet_latent_3d if dim == 3 else args.tiny_deeponet_latent
        cno_levels = args.tiny_cno_levels_3d if dim == 3 else args.tiny_cno_levels
    else:
        width = args.full_width_3d if dim == 3 else args.full_width
        layers = args.full_layers
        modes = args.full_modes_3d if dim == 3 else args.full_modes
        latent = args.full_deeponet_latent_3d if dim == 3 else args.full_deeponet_latent
        cno_levels = args.full_cno_levels_3d if dim == 3 else args.full_cno_levels
    return {
        "width": int(width),
        "layers": int(layers),
        "modes": int(modes),
        "latent": int(latent),
        "cno_levels": int(cno_levels),
    }


def build_operator(family: str, problem: LoadedProblem, args: argparse.Namespace, *, tiny: bool) -> nn.Module:
    hp = full_tiny_hparams(problem.dim, args, tiny=tiny)
    min_spatial = min(problem.spatial_shape)
    modes = max(2, min(hp["modes"], max(2, min_spatial // 2)))
    if family == "fno":
        return FNOOperator(problem.dim, problem.cin, problem.cout, width=hp["width"], modes=modes, layers=hp["layers"], dropout=args.dropout)
    if family == "deeponet":
        branch_levels = max(2, min(hp["layers"], int(math.floor(math.log2(max(4, min_spatial))))))
        return DeepONetOperator(
            problem.dim,
            problem.cin,
            problem.cout,
            width=hp["width"],
            latent=hp["latent"],
            spatial_shape=problem.spatial_shape,
            branch_levels=branch_levels,
            trunk_layers=max(3, hp["layers"]),
            fourier_frequencies=args.deeponet_fourier_features,
            dropout=args.dropout,
        )
    if family == "cno":
        max_levels = max(2, int(math.floor(math.log2(max(4, min_spatial)))) - 1)
        levels = max(2, min(hp["cno_levels"], max_levels))
        return CNOOperator(
            problem.dim,
            problem.cin,
            problem.cout,
            width=hp["width"],
            levels=levels,
            blocks_per_level=args.cno_blocks_per_level,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown family {family}")


# =============================================================================
# Training / inference
# =============================================================================

def spectral_mse(pred: torch.Tensor, target: torch.Tensor, dim: int, s: float = 1.0) -> torch.Tensor:
    err = pred - target
    z = channels_last_to_first(err, dim)
    spatial = z.shape[2:]
    zhat = torch.fft.rfftn(z, dim=tuple(range(2, z.ndim)), norm="ortho")
    axes = []
    for ax, n in enumerate(spatial):
        if ax == len(spatial) - 1:
            axes.append(torch.fft.rfftfreq(n, d=1.0 / n).to(err.device))
        else:
            axes.append(torch.fft.fftfreq(n, d=1.0 / n).to(err.device))
    mesh = torch.meshgrid(*axes, indexing="ij")
    kk2 = sum(g ** 2 for g in mesh)
    wt = (1.0 + kk2) ** (0.5 * s)
    view_shape = (1, 1, *wt.shape)
    return ((zhat.abs() * wt.view(*view_shape)) ** 2).mean() / max(1, int(np.prod(spatial)))


def make_loss_fn(dim: int, spectral_beta: float, spectral_s: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, target)
        if spectral_beta > 0.0:
            loss = loss + spectral_beta * spectral_mse(pred, target, dim=dim, s=spectral_s)
        return loss
    return loss_fn


@torch.no_grad()
def predict_batches(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model = model.to(device)
    model.eval()
    preds = []
    for i in range(0, x.shape[0], batch_size):
        preds.append(model(x[i : i + batch_size].to(device)).cpu())
    return torch.cat(preds, dim=0)


@torch.no_grad()
def eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], batch_size: int, device: torch.device) -> float:
    model = model.to(device)
    model.eval()
    vals: List[float] = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].to(device)
        yb = y[i : i + batch_size].to(device)
        vals.append(float(loss_fn(model(xb), yb).detach().cpu()))
    return float(np.mean(vals)) if vals else float("nan")


def train_fixed_epochs(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    grad_clip: float,
    label: str,
    verbose: bool,
) -> Tuple[nn.Module, Dict[str, float]]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    best_val = float("inf")
    best_epoch = 0
    final_train = float("nan")
    final_val = float("nan")
    for ep in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        final_train = float(np.mean(losses)) if losses else float("nan")
        final_val = eval_loss(model, x_val, y_val, loss_fn, batch_size, device)
        if final_val < best_val:
            best_val = final_val
            best_epoch = ep
        if verbose or ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"      {label} epoch {ep:03d}/{epochs} | train={final_train:.4e} | val={final_val:.4e} | best_seen={best_val:.4e}@{best_epoch}")
    model = model.cpu()
    return model, {
        "epochs_run": int(epochs),
        "final_train_loss": float(final_train),
        "final_val_loss": float(final_val),
        "best_val_loss_seen": float(best_val),
        "best_val_epoch_seen": int(best_epoch),
    }


def choose_eta(y_val: torch.Tensor, current_val: torch.Tensor, corr_val: torch.Tensor, candidates: Sequence[float]) -> Tuple[float, float]:
    best_eta = float(candidates[0])
    best_loss = float("inf")
    for eta in candidates:
        loss = F.mse_loss(current_val + float(eta) * corr_val, y_val).item()
        if loss < best_loss:
            best_loss = loss
            best_eta = float(eta)
    return best_eta, best_loss


def eval_denorm_rel(pred_norm: torch.Tensor, problem: LoadedProblem) -> float:
    pred_raw = problem.denorm_y(pred_norm)
    return float(rel_l2(pred_raw, problem.y_test_raw).mean())


# =============================================================================
# Experiment core
# =============================================================================

def run_family_dataset_seed(family: str, problem: LoadedProblem, seed: int, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    set_seed(seed)
    batch_size = args.batch_size_3d if problem.dim == 3 else args.batch_size
    lr = args.lr_3d if problem.dim == 3 else args.lr

    print(f"\n--- family={family} | seed={seed} | dataset={problem.name} | dim={problem.dim} | train={problem.x_train.shape[0]} val={problem.x_val.shape[0]} test={problem.x_test.shape[0]} ---")

    full = build_operator(family, problem, args, tiny=False)
    full_params = count_params(full)
    print(f"  training FULL {family}: params={full_params:,}")
    full_loss = make_loss_fn(problem.dim, spectral_beta=args.full_spectral_beta, spectral_s=args.spectral_s)
    full, full_stats = train_fixed_epochs(
        full,
        problem.x_train,
        problem.y_train,
        problem.x_val,
        problem.y_val,
        loss_fn=full_loss,
        epochs=args.full_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=args.weight_decay,
        device=device,
        grad_clip=args.grad_clip,
        label=f"full/{family}",
        verbose=args.verbose,
    )
    full_pred_test = predict_batches(full, problem.x_test, batch_size, device)
    full_rel = eval_denorm_rel(full_pred_test, problem)
    full_val_norm_rel = float(rel_l2(predict_batches(full, problem.x_val, batch_size, device), problem.y_val).mean())
    print(f"  FULL {family}: val_rel_norm={full_val_norm_rel:.4f} | test_rel_l2={full_rel:.6f}")

    # Tiny boosted ensemble from zero / normalized mean predictor.
    pred_train = torch.zeros_like(problem.y_train)
    pred_val = torch.zeros_like(problem.y_val)
    pred_test = torch.zeros_like(problem.y_test)
    residual_loss = make_loss_fn(problem.dim, spectral_beta=args.boost_spectral_beta, spectral_s=args.spectral_s)
    etas: List[float] = []
    stage_rows: List[Dict[str, Any]] = []
    tiny_params_single: Optional[int] = None
    best_boost_rel = eval_denorm_rel(pred_test, problem)

    for m in range(args.stages):
        residual_train = problem.y_train - pred_train
        residual_val = problem.y_val - pred_val
        train_resid_rel = float(rel_l2(residual_train, problem.y_train).mean())
        val_resid_rel = float(rel_l2(residual_val, problem.y_val).mean())
        tiny = build_operator(family, problem, args, tiny=True)
        tiny_params = count_params(tiny)
        if tiny_params_single is None:
            tiny_params_single = tiny_params
        print(f"  boosting stage {m + 1}/{args.stages} tiny/{family}: params={tiny_params:,} | train_resid_rel={train_resid_rel:.4f} val_resid_rel={val_resid_rel:.4f}")
        tiny, tiny_stats = train_fixed_epochs(
            tiny,
            problem.x_train,
            residual_train,
            problem.x_val,
            residual_val,
            loss_fn=residual_loss,
            epochs=args.tiny_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=args.weight_decay,
            device=device,
            grad_clip=args.grad_clip,
            label=f"tiny/{family}/H{m + 1}",
            verbose=args.verbose,
        )
        corr_train = predict_batches(tiny, problem.x_train, batch_size, device)
        corr_val = predict_batches(tiny, problem.x_val, batch_size, device)
        corr_test = predict_batches(tiny, problem.x_test, batch_size, device)
        eta, eta_val_mse = choose_eta(problem.y_val, pred_val, corr_val, args.eta_candidates)
        etas.append(float(eta))
        pred_train = pred_train + eta * corr_train
        pred_val = pred_val + eta * corr_val
        pred_test = pred_test + eta * corr_test
        boost_val_rel = float(rel_l2(pred_val, problem.y_val).mean())
        boost_test_rel = eval_denorm_rel(pred_test, problem)
        best_boost_rel = min(best_boost_rel, boost_test_rel)
        stage_rows.append({
            "stage": int(m + 1),
            "eta": float(eta),
            "eta_val_mse": float(eta_val_mse),
            "boost_val_rel_l2_norm": float(boost_val_rel),
            "boost_test_rel_l2": float(boost_test_rel),
            "tiny_params": int(tiny_params),
            **{f"tiny_{k}": v for k, v in tiny_stats.items()},
        })
        print(f"    eta={eta:.3f} | boosted_val_rel_norm={boost_val_rel:.4f} | boosted_test_rel_l2={boost_test_rel:.6f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    boost_rel = eval_denorm_rel(pred_test, problem)
    total_tiny_params = int((tiny_params_single or 0) * args.stages)
    improvement = pct_improvement(full_rel, boost_rel)
    best_improvement = pct_improvement(full_rel, best_boost_rel)
    print(f"  RESULT {family}/{problem.name}/seed={seed}: full={full_rel:.6f}, boosted_tiny={boost_rel:.6f}, improvement={improvement:.2f}%")

    return {
        "family": family,
        "dataset": problem.name,
        "label": problem.label,
        "source": problem.source,
        "dim": int(problem.dim),
        "group": problem.group,
        "seed": int(seed),
        "full_rel_l2": float(full_rel),
        "boosted_tiny_rel_l2": float(boost_rel),
        "best_stage_boosted_tiny_rel_l2": float(best_boost_rel),
        "improvement_pct": float(improvement),
        "best_stage_improvement_pct": float(best_improvement),
        "full_val_rel_l2_norm": float(full_val_norm_rel),
        "full_params": int(full_params),
        "tiny_single_params": int(tiny_params_single or 0),
        "boosted_tiny_total_params": int(total_tiny_params),
        "boosted_to_full_param_ratio": float(total_tiny_params / max(1, full_params)),
        "etas": [float(e) for e in etas],
        "etas_mean": float(np.mean(etas)) if etas else 0.0,
        "full_train_stats": full_stats,
        "stage_metrics": stage_rows,
    }


# =============================================================================
# Aggregation
# =============================================================================

def bootstrap_mean_ci(values: Sequence[float], *, n_boot: int, ci: float, seed: int) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci_level": float(ci), "ci_low": float("nan"), "ci_high": float("nan"), "values": []}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boot[b] = np.mean(rng.choice(arr, size=arr.size, replace=True))
    alpha = 1.0 - ci
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "ci_level": float(ci),
        "ci_low": float(np.percentile(boot, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0))),
        "values": [float(v) for v in arr.tolist()],
    }


def aggregate_rows(rows: List[Dict[str, Any]], *, n_boot: int, ci: float, seed: int) -> Dict[str, Any]:
    ok = [r for r in rows if "improvement_pct" in r and np.isfinite(float(r["improvement_pct"]))]
    out: Dict[str, Any] = {}

    def group_stats(keys: Sequence[str], offset: int) -> Dict[str, Any]:
        groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for r in ok:
            k = tuple(r[key] for key in keys)
            groups.setdefault(k, []).append(r)
        res: Dict[str, Any] = {}
        for j, (k, vals) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
            label = "/".join(str(x) for x in k)
            res[label] = {
                "keys": list(k),
                "improvement_pct": bootstrap_mean_ci([float(v["improvement_pct"]) for v in vals], n_boot=n_boot, ci=ci, seed=seed + offset + 10 * j + 1),
                "best_stage_improvement_pct": bootstrap_mean_ci([float(v["best_stage_improvement_pct"]) for v in vals], n_boot=n_boot, ci=ci, seed=seed + offset + 10 * j + 2),
                "full_rel_l2": bootstrap_mean_ci([float(v["full_rel_l2"]) for v in vals], n_boot=n_boot, ci=ci, seed=seed + offset + 10 * j + 3),
                "boosted_tiny_rel_l2": bootstrap_mean_ci([float(v["boosted_tiny_rel_l2"]) for v in vals], n_boot=n_boot, ci=ci, seed=seed + offset + 10 * j + 4),
                "boosted_to_full_param_ratio": bootstrap_mean_ci([float(v["boosted_to_full_param_ratio"]) for v in vals], n_boot=n_boot, ci=ci, seed=seed + offset + 10 * j + 5),
            }
        return res

    out["by_family_dataset"] = group_stats(["family", "dataset"], 1000)
    out["by_family"] = group_stats(["family"], 2000)
    out["by_family_dim"] = group_stats(["family", "group"], 3000)
    out["by_dim"] = group_stats(["group"], 4000)

    # Macro by seed/family/dataset: for each seed average across all completed family-dataset pairs.
    seed_to_vals: Dict[int, List[float]] = {}
    for r in ok:
        seed_to_vals.setdefault(int(r["seed"]), []).append(float(r["improvement_pct"]))
    macro_by_seed = [float(np.mean(v)) for _, v in sorted(seed_to_vals.items()) if v]
    pooled = [float(r["improvement_pct"]) for r in ok]
    out["macro_average_by_seed"] = {"improvement_pct": bootstrap_mean_ci(macro_by_seed, n_boot=n_boot, ci=ci, seed=seed + 9001)}
    out["pooled_family_dataset_seed"] = {"improvement_pct": bootstrap_mean_ci(pooled, n_boot=n_boot, ci=ci, seed=seed + 9002)}
    return out


def print_summary(aggregate: Dict[str, Any]) -> None:
    print("\n\n=== Full-size baseline vs boosted tiny ensemble: percent improvement ===")
    print("Positive means boosted tiny ensemble beats full-size standalone baseline.")
    print("\nBy family/dataset")
    print("Family/Dataset                              mean     95% CI     n")
    print("-" * 78)
    for key, block in aggregate.get("by_family_dataset", {}).items():
        s = block["improvement_pct"]
        print(f"{key:<40} {s['mean']:>7.2f}%  [{s['ci_low']:>7.2f}, {s['ci_high']:>7.2f}]  {s['n']:>3}")
    print("\nBy family")
    print("Family                                    mean     95% CI     n")
    print("-" * 78)
    for key, block in aggregate.get("by_family", {}).items():
        s = block["improvement_pct"]
        print(f"{key:<40} {s['mean']:>7.2f}%  [{s['ci_low']:>7.2f}, {s['ci_high']:>7.2f}]  {s['n']:>3}")
    macro = aggregate.get("macro_average_by_seed", {}).get("improvement_pct", {})
    pooled = aggregate.get("pooled_family_dataset_seed", {}).get("improvement_pct", {})
    print("-" * 78)
    if macro:
        print(f"{'macro avg by seed':<40} {macro['mean']:>7.2f}%  [{macro['ci_low']:>7.2f}, {macro['ci_high']:>7.2f}]  {macro['n']:>3}")
    if pooled:
        print(f"{'pooled family-dataset-seed':<40} {pooled['mean']:>7.2f}%  [{pooled['ci_low']:>7.2f}, {pooled['ci_high']:>7.2f}]  {pooled['n']:>3}")


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["etas"] = json.dumps(rr.get("etas", []))
        rr.pop("stage_metrics", None)
        rr.pop("full_train_stats", None)
        flat_rows.append(rr)
    if not flat_rows:
        return
    keys = sorted({k for r in flat_rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in flat_rows:
            writer.writerow(r)



# =============================================================================
# Resume / checkpoint helpers
# =============================================================================

def _row_key(row: Dict[str, Any]) -> Optional[Tuple[str, int, str]]:
    try:
        return (str(row["dataset"]), int(row["seed"]), str(row["family"]))
    except Exception:
        return None


def _coerce_scalar(value: Any) -> Any:
    """Best-effort scalar coercion for rows loaded from CSV fallback."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s == "":
        return value
    # JSON lists/dicts such as etas.
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            return value
    try:
        if any(ch in s for ch in [".", "e", "E", "nan", "inf"]):
            return float(s)
        return int(s)
    except Exception:
        return value


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the last complete row for each dataset/seed/family key."""
    ordered: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    passthrough: List[Dict[str, Any]] = []
    for row in rows:
        key = _row_key(row)
        if key is None:
            passthrough.append(row)
        else:
            ordered[key] = row
    return passthrough + list(ordered.values())


def load_resume_state(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load prior rows/errors so interrupted runs skip completed jobs."""
    if getattr(args, "no_resume", False):
        return [], []

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if args.results.exists():
        try:
            with open(args.results, "r") as f:
                payload = json.load(f)
            rows = list(payload.get("rows", []) or [])
            errors = list(payload.get("errors", []) or [])
            rows = _dedupe_rows(rows)
            print(f"[RESUME] Loaded {len(rows)} completed row(s) and {len(errors)} prior error(s) from {args.results}")
            return rows, errors
        except Exception as e:
            print(f"[WARN] Could not load resume JSON {args.results}: {e}")

    # CSV fallback: enough to skip completed work if JSON is unavailable.
    if args.csv.exists():
        try:
            with open(args.csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k: _coerce_scalar(v) for k, v in row.items()})
            rows = _dedupe_rows(rows)
            print(f"[RESUME] Loaded {len(rows)} completed row(s) from CSV fallback {args.csv}")
        except Exception as e:
            print(f"[WARN] Could not load resume CSV {args.csv}: {e}")

    return rows, errors


def completed_keys(rows: List[Dict[str, Any]]) -> set[Tuple[str, int, str]]:
    keys: set[Tuple[str, int, str]] = set()
    for row in rows:
        key = _row_key(row)
        if key is not None:
            keys.add(key)
    return keys


def save_checkpoint(
    args: argparse.Namespace,
    start_time: float,
    all_rows: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
) -> None:
    """Persist partial results after each completed family run."""
    aggregate = aggregate_rows(all_rows, n_boot=args.bootstrap_samples, ci=args.ci, seed=args.bootstrap_seed) if all_rows else {}
    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": all_rows,
        "errors": errors,
        "aggregate": aggregate,
        "elapsed_seconds": float(time.time() - start_time),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.results.with_suffix(args.results.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, args.results)
    write_csv(all_rows, args.csv)

# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-size neural operators vs boosted tiny same-family ensembles")
    p.add_argument("--common-data-root", type=Path, default=Path("data"), help="Root containing current standardized data/common/*.h5")
    p.add_argument("--external-data-root", type=Path, default=Path.home() / "Desktop/research_work/COMET/comet_ncs_codebase/data/raw")
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    p.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES, choices=DEFAULT_FAMILIES)
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Data budgets.
    p.add_argument("--max-samples", type=int, default=256, help="Max samples for standardized data/common tasks")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-train-pairs", type=int, default=1024)
    p.add_argument("--max-val-pairs", type=int, default=256)
    p.add_argument("--max-test-pairs", type=int, default=256)
    p.add_argument("--max-delta", type=int, default=4)
    p.add_argument("--spatial-limit-1d", type=int, default=0)
    p.add_argument("--spatial-limit-2d", type=int, default=64)
    p.add_argument("--spatial-limit-3d", type=int, default=32)

    # Training.
    p.add_argument("--full-epochs", type=int, default=100)
    p.add_argument("--tiny-epochs", type=int, default=100)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batch-size-3d", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-3d", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)

    # Full-size model budgets.
    p.add_argument("--full-width", type=int, default=64)
    p.add_argument("--full-width-3d", type=int, default=32)
    p.add_argument("--full-layers", type=int, default=6)
    p.add_argument("--full-modes", type=int, default=16)
    p.add_argument("--full-modes-3d", type=int, default=8)
    p.add_argument("--full-deeponet-latent", type=int, default=256)
    p.add_argument("--full-deeponet-latent-3d", type=int, default=128)
    p.add_argument("--full-cno-levels", type=int, default=4)
    p.add_argument("--full-cno-levels-3d", type=int, default=3)

    # Tiny boosted model budgets.
    p.add_argument("--tiny-width", type=int, default=24)
    p.add_argument("--tiny-width-3d", type=int, default=16)
    p.add_argument("--tiny-layers", type=int, default=3)
    p.add_argument("--tiny-modes", type=int, default=8)
    p.add_argument("--tiny-modes-3d", type=int, default=4)
    p.add_argument("--tiny-deeponet-latent", type=int, default=64)
    p.add_argument("--tiny-deeponet-latent-3d", type=int, default=48)
    p.add_argument("--tiny-cno-levels", type=int, default=3)
    p.add_argument("--tiny-cno-levels-3d", type=int, default=2)

    # Family-specific.
    p.add_argument("--deeponet-fourier-features", type=int, default=8)
    p.add_argument("--cno-blocks-per-level", type=int, default=2)

    # Boosting.
    p.add_argument("--full-spectral-beta", type=float, default=0.0)
    p.add_argument("--boost-spectral-beta", type=float, default=0.02)
    p.add_argument("--spectral-s", type=float, default=1.0)
    p.add_argument("--eta-candidates", nargs="*", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3])

    # Aggregation / output.
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--results", type=Path, default=Path("results_full_vs_boosted_tiny_10seeds.json"))
    p.add_argument("--csv", type=Path, default=Path("results_full_vs_boosted_tiny_10seeds.csv"))
    p.add_argument("--no-resume", action="store_true", help="Start a fresh run even if existing results JSON/CSV files are present")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--continue-on-error", action="store_true", default=True)
    p.add_argument("--smoke", action="store_true", help="Tiny run for loader/model sanity checks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seeds = args.seeds[:2]
        args.datasets = args.datasets[:2]
        args.full_epochs = min(args.full_epochs, 2)
        args.tiny_epochs = min(args.tiny_epochs, 2)
        args.stages = min(args.stages, 2)
        args.max_samples = min(args.max_samples, 32)
        args.max_train_pairs = min(args.max_train_pairs, 64)
        args.max_val_pairs = min(args.max_val_pairs, 32)
        args.max_test_pairs = min(args.max_test_pairs, 32)
        args.full_width = min(args.full_width, 16)
        args.tiny_width = min(args.tiny_width, 8)
        args.full_width_3d = min(args.full_width_3d, 12)
        args.tiny_width_3d = min(args.tiny_width_3d, 8)
        args.full_deeponet_latent = min(args.full_deeponet_latent, 32)
        args.tiny_deeponet_latent = min(args.tiny_deeponet_latent, 16)

    if len(args.seeds) < 2:
        raise ValueError("Use at least two seeds; default is ten seeds.")

    device = torch.device(args.device)
    start_time = time.time()
    all_rows, errors = load_resume_state(args)
    done = completed_keys(all_rows)

    if args.no_resume:
        print("[RESUME] Disabled by --no-resume; starting fresh and overwriting outputs.")
    elif done:
        print(f"[RESUME] Will skip {len(done)} completed dataset/seed/family job(s).")

    for dataset in args.datasets:
        for seed in args.seeds:
            pending_families = [family for family in args.families if (dataset, int(seed), family) not in done]
            if not pending_families:
                print(f"\n[SKIP] dataset={dataset} seed={seed}: all requested families already complete")
                continue

            try:
                problem = load_problem(dataset, args, seed)
                print(f"\n=== loaded dataset={dataset} seed={seed} | source={problem.source} dim={problem.dim} shape={problem.spatial_shape} cin={problem.cin} cout={problem.cout} ===")
                for family in args.families:
                    key = (dataset, int(seed), family)
                    if key in done:
                        print(f"  [SKIP] family={family} | seed={seed} | dataset={dataset}: already complete")
                        continue
                    try:
                        row = run_family_dataset_seed(family, problem, seed, args, device)
                        all_rows.append(row)
                        done.add(key)
                        save_checkpoint(args, start_time, all_rows, errors)
                        print(f"  [CHECKPOINT] saved after dataset={dataset} seed={seed} family={family}")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    except Exception as e:
                        err = {"dataset": dataset, "seed": int(seed), "family": family, "error": repr(e)}
                        errors.append(err)
                        save_checkpoint(args, start_time, all_rows, errors)
                        print(f"[ERROR] {err}")
                        if not args.continue_on_error:
                            raise
            except Exception as e:
                err = {"dataset": dataset, "seed": int(seed), "family": "ALL", "error": repr(e)}
                errors.append(err)
                save_checkpoint(args, start_time, all_rows, errors)
                print(f"[ERROR] {err}")
                if not args.continue_on_error:
                    raise

    # Final write and summary.
    save_checkpoint(args, start_time, all_rows, errors)
    aggregate = aggregate_rows(all_rows, n_boot=args.bootstrap_samples, ci=args.ci, seed=args.bootstrap_seed) if all_rows else {}
    if aggregate:
        print_summary(aggregate)
    print(f"\nwrote {args.results}")
    print(f"wrote {args.csv}")
    if errors:
        print(f"completed with {len(errors)} errors; inspect JSON `errors` field")


if __name__ == "__main__":
    main()
