#!/usr/bin/env python3
"""
download_pdebench.py

Common data builder/downloader for toy Randomized Local Neural Operators (RLNO.py)
and Functional Gradient Boosted Neural Operators (GBNO.py).

Default behavior creates lightweight synthetic 1D PDE datasets and synthetic 2D
PDEBench-shaped proxy datasets in a common standardized HDF5 format:

    data/common/<dataset>.h5
        a: [N, spatial..., input_channels]
        u: [N, spatial..., output_channels]

Optional behavior downloads selected official PDEBench 2D files from DaRUS and
standardizes small subsets into the same format.

Official PDEBench 2D files are large. The PDEBench README currently lists Darcy
as ~6.2 GB and 2D reaction-diffusion as ~13 GB. Use --download-pdebench only
when you really want those downloads.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class PDEBenchFile:
    name: str
    filename: str
    file_id: int
    standard_name: str
    approx_size: str

    @property
    def url(self) -> str:
        return f"https://darus.uni-stuttgart.de/api/access/datafile/{self.file_id}"


PDEBENCH_2D: Dict[str, PDEBenchFile] = {
    "darcy": PDEBenchFile(
        name="darcy",
        filename="2D_DarcyFlow_beta1.0_Train.hdf5",
        file_id=133219,
        standard_name="pdebench_darcy",
        approx_size="6.2 GB",
    ),
    "2d_reacdiff": PDEBenchFile(
        name="2d_reacdiff",
        filename="2D_diff-react_NA_NA.h5",
        file_id=133017,
        standard_name="pdebench_2d_reacdiff",
        approx_size="13 GB",
    ),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_standard_h5(
    out_path: Path,
    a: np.ndarray,
    u: np.ndarray,
    *,
    name: str,
    source: str,
    equation: str,
) -> None:
    ensure_dir(out_path.parent)
    a = np.asarray(a, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    assert a.shape[0] == u.shape[0], (a.shape, u.shape)
    assert a.ndim == u.ndim, (a.shape, u.shape)
    if a.ndim < 3:
        raise ValueError(f"Expected [N, spatial..., C], got a.shape={a.shape}")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("a", data=a, compression="gzip", compression_opts=4)
        f.create_dataset("u", data=u, compression="gzip", compression_opts=4)
        f.attrs["name"] = name
        f.attrs["source"] = source
        f.attrs["equation"] = equation
        f.attrs["ndim"] = a.ndim - 2
        f.attrs["input_channels"] = a.shape[-1]
        f.attrs["output_channels"] = u.shape[-1]
    print(f"wrote {out_path} | a={a.shape}, u={u.shape}, source={source}")


def random_periodic_ic(rng: np.random.Generator, n: int, nx: int, modes: int = 6) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    out = np.zeros((n, nx), dtype=np.float64)
    for i in range(n):
        coeff_decay = 1.0 / np.arange(1, modes + 1) ** 1.3
        for k in range(1, modes + 1):
            amp = rng.normal() * coeff_decay[k - 1]
            phase = rng.uniform(0.0, 2.0 * np.pi)
            out[i] += amp * np.sin(2.0 * np.pi * k * x + phase)
        out[i] -= out[i].mean()
        std = out[i].std() + 1e-6
        out[i] /= std
    return out.astype(np.float32)


def spectral_derivatives_1d(u: np.ndarray, length: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return ux and uxx for periodic grid using FFT. u: [N, X]."""
    nx = u.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    uhat = np.fft.fft(u, axis=-1)
    ux = np.fft.ifft(1j * k * uhat, axis=-1).real
    uxx = np.fft.ifft(-(k**2) * uhat, axis=-1).real
    return ux, uxx


def make_1d_advection(out_dir: Path, n: int, nx: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    u0 = random_periodic_ic(rng, n, nx)
    c = rng.uniform(0.25, 1.5, size=(n, 1)).astype(np.float32)
    t_final = 0.25
    freqs = np.fft.fftfreq(nx, d=1.0 / nx)
    u0hat = np.fft.fft(u0, axis=-1)
    phase = np.exp(-2j * np.pi * freqs[None, :] * c * t_final)
    uT = np.fft.ifft(u0hat * phase, axis=-1).real.astype(np.float32)
    c_chan = np.broadcast_to(c[:, None, :], (n, nx, 1))
    a = np.concatenate([u0[..., None], c_chan], axis=-1)
    u = uT[..., None]
    write_standard_h5(out_dir / "1d_advection.h5", a, u, name="1d_advection", source="synthetic", equation="u_t + c u_x = 0")


def make_1d_burgers(out_dir: Path, n: int, nx: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    u = random_periodic_ic(rng, n, nx)
    nu = rng.uniform(0.005, 0.05, size=(n, 1)).astype(np.float64)
    u0 = u.copy()
    dt = 1.0e-3
    steps = 100

    def rhs(v: np.ndarray) -> np.ndarray:
        vx, vxx = spectral_derivatives_1d(v)
        return -v * vx + nu * vxx

    for _ in range(steps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u = np.clip(u, -10.0, 10.0)
    nu_chan = np.broadcast_to(nu.astype(np.float32)[:, None, :], (n, nx, 1))
    a = np.concatenate([u0[..., None].astype(np.float32), nu_chan], axis=-1)
    y = u[..., None].astype(np.float32)
    write_standard_h5(out_dir / "1d_burgers.h5", a, y, name="1d_burgers", source="synthetic", equation="u_t + u u_x = nu u_xx")


def make_1d_reacdiff(out_dir: Path, n: int, nx: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    raw = random_periodic_ic(rng, n, nx)
    u = 1.0 / (1.0 + np.exp(-raw))
    D = rng.uniform(0.002, 0.025, size=(n, 1)).astype(np.float64)
    rho = rng.uniform(0.5, 2.0, size=(n, 1)).astype(np.float64)
    u0 = u.copy()
    dt = 1.0e-3
    steps = 150
    for _ in range(steps):
        _, uxx = spectral_derivatives_1d(u)
        u = u + dt * (D * uxx + rho * u * (1.0 - u))
        u = np.clip(u, 0.0, 2.0)
    D_chan = np.broadcast_to(D.astype(np.float32)[:, None, :], (n, nx, 1))
    rho_chan = np.broadcast_to(rho.astype(np.float32)[:, None, :], (n, nx, 1))
    a = np.concatenate([u0[..., None].astype(np.float32), D_chan, rho_chan], axis=-1)
    y = u[..., None].astype(np.float32)
    write_standard_h5(out_dir / "1d_reacdiff.h5", a, y, name="1d_reacdiff", source="synthetic", equation="u_t = D u_xx + rho u(1-u)")


def smooth_random_field_2d(rng: np.random.Generator, n: int, res: int, cutoff: float = 7.0) -> np.ndarray:
    noise = rng.normal(size=(n, res, res))
    kh = np.fft.fftfreq(res)[:, None]
    kw = np.fft.fftfreq(res)[None, :]
    filt = np.exp(-0.5 * ((kh**2 + kw**2) * (res / cutoff) ** 2))
    out = np.fft.ifft2(np.fft.fft2(noise, axes=(-2, -1)) * filt, axes=(-2, -1)).real
    out -= out.mean(axis=(-2, -1), keepdims=True)
    out /= out.std(axis=(-2, -1), keepdims=True) + 1e-6
    return out.astype(np.float32)


def poisson_solve_periodic(rhs: np.ndarray) -> np.ndarray:
    """Solve -Delta u = rhs on periodic square with zero-mean compatibility."""
    n, h, w = rhs.shape
    rhs = rhs - rhs.mean(axis=(-2, -1), keepdims=True)
    kx = 2.0 * np.pi * np.fft.fftfreq(h, d=1.0 / h)
    ky = 2.0 * np.pi * np.fft.fftfreq(w, d=1.0 / w)
    kk = kx[:, None] ** 2 + ky[None, :] ** 2
    kk[0, 0] = 1.0
    uhat = np.fft.fft2(rhs, axes=(-2, -1)) / kk[None, :, :]
    uhat[:, 0, 0] = 0.0
    return np.fft.ifft2(uhat, axes=(-2, -1)).real.astype(np.float32)


def make_2d_darcy_proxy(out_dir: Path, n: int, res: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    log_k = smooth_random_field_2d(rng, n, res)
    forcing = smooth_random_field_2d(rng, n, res, cutoff=4.0)
    k = np.exp(0.7 * log_k)
    # Lightweight proxy: variable coefficient modulates effective RHS before a Poisson solve.
    u = poisson_solve_periodic(forcing / (0.2 + k))
    a = np.stack([k, forcing], axis=-1).astype(np.float32)
    y = u[..., None]
    write_standard_h5(
        out_dir / "pdebench_darcy.h5",
        a,
        y,
        name="pdebench_darcy",
        source="synthetic_proxy_for_pdebench_darcy",
        equation="Darcy-like periodic Poisson proxy",
    )


def laplacian_2d_periodic(v: np.ndarray) -> np.ndarray:
    return (
        np.roll(v, 1, axis=-2)
        + np.roll(v, -1, axis=-2)
        + np.roll(v, 1, axis=-1)
        + np.roll(v, -1, axis=-1)
        - 4.0 * v
    )


def make_2d_reacdiff_proxy(out_dir: Path, n: int, res: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    u = 0.5 + 0.08 * smooth_random_field_2d(rng, n, res, cutoff=5.0)
    v = 0.25 + 0.08 * smooth_random_field_2d(rng, n, res, cutoff=5.0)
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    init = np.stack([u, v], axis=-1).astype(np.float32)
    Du, Dv = 0.12, 0.06
    feed = rng.uniform(0.025, 0.055, size=(n, 1, 1))
    kill = rng.uniform(0.045, 0.075, size=(n, 1, 1))
    dt = 0.15
    for _ in range(60):
        uvv = u * v * v
        u = u + dt * (Du * laplacian_2d_periodic(u) - uvv + feed * (1.0 - u))
        v = v + dt * (Dv * laplacian_2d_periodic(v) + uvv - (feed + kill) * v)
        u = np.clip(u, -0.2, 1.5)
        v = np.clip(v, -0.2, 1.5)
    y = np.stack([u, v], axis=-1).astype(np.float32)
    feed_chan = np.broadcast_to(feed[..., None].astype(np.float32), (n, res, res, 1))
    kill_chan = np.broadcast_to(kill[..., None].astype(np.float32), (n, res, res, 1))
    a = np.concatenate([init, feed_chan, kill_chan], axis=-1)
    write_standard_h5(
        out_dir / "pdebench_2d_reacdiff.h5",
        a,
        y,
        name="pdebench_2d_reacdiff",
        source="synthetic_proxy_for_pdebench_2d_reacdiff",
        equation="Gray-Scott-like 2D reaction-diffusion proxy",
    )


def download_file(url: str, out_path: Path, overwrite: bool = False) -> None:
    ensure_dir(out_path.parent)
    if out_path.exists() and not overwrite:
        print(f"exists, skipping: {out_path}")
        return
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    print(f"downloading {url}\n  -> {out_path}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        total = response.headers.get("Content-Length")
        total_i = int(total) if total is not None else None
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total_i:
                pct = 100.0 * done / total_i
                print(f"\r  {done / 1e9:.2f}/{total_i / 1e9:.2f} GB ({pct:.1f}%)", end="")
        print()
    tmp.replace(out_path)


def downsample_spatial(x: np.ndarray, target: Optional[int]) -> np.ndarray:
    if target is None:
        return x
    if x.ndim == 3:  # [N, X, C]
        stride = max(1, x.shape[1] // target)
        return x[:, ::stride, :][:, :target, :]
    if x.ndim == 4:  # [N, H, W, C]
        sh = max(1, x.shape[1] // target)
        sw = max(1, x.shape[2] // target)
        return x[:, ::sh, ::sw, :][:, :target, :target, :]
    return x


def standardize_pdebench_file(
    name: str,
    raw_path: Path,
    out_dir: Path,
    max_samples: int,
    target_grid: Optional[int],
) -> None:
    info = PDEBENCH_2D[name]
    if name == "darcy":
        with h5py.File(raw_path, "r") as f:
            if "nu" not in f or "tensor" not in f:
                raise KeyError(f"Expected keys 'nu' and 'tensor' in Darcy file. Found {list(f.keys())}")
            a = np.asarray(f["nu"][:max_samples], dtype=np.float32)
            u = np.asarray(f["tensor"][:max_samples], dtype=np.float32)
        while a.ndim < 4:
            a = a[..., None]
        while u.ndim < 4:
            u = u[..., None]
        a = downsample_spatial(a, target_grid)
        u = downsample_spatial(u, target_grid)
        write_standard_h5(out_dir / f"{info.standard_name}.h5", a, u, name=info.standard_name, source="PDEBench_DaRUS_subset", equation="2D Darcy flow")
        return

    if name == "2d_reacdiff":
        a_list: List[np.ndarray] = []
        u_list: List[np.ndarray] = []
        with h5py.File(raw_path, "r") as f:
            keys = sorted(list(f.keys()))[:max_samples]
            for key in keys:
                data = np.asarray(f[f"{key}/data"], dtype=np.float32)  # [T, H, W, C]
                a_list.append(data[0])
                u_list.append(data[-1])
        a = np.stack(a_list, axis=0)
        u = np.stack(u_list, axis=0)
        a = downsample_spatial(a, target_grid)
        u = downsample_spatial(u, target_grid)
        write_standard_h5(out_dir / f"{info.standard_name}.h5", a, u, name=info.standard_name, source="PDEBench_DaRUS_subset", equation="2D reaction-diffusion")
        return

    raise ValueError(f"Unsupported PDEBench dataset: {name}")


def make_synthetic_all(out_dir: Path, n1d: int, n2d: int, nx: int, res2d: int, seed: int) -> None:
    ensure_dir(out_dir)
    make_1d_advection(out_dir, n1d, nx, seed + 1)
    make_1d_burgers(out_dir, n1d, nx, seed + 2)
    make_1d_reacdiff(out_dir, n1d, nx, seed + 3)
    make_2d_darcy_proxy(out_dir, n2d, res2d, seed + 4)
    make_2d_reacdiff_proxy(out_dir, n2d, res2d, seed + 5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/download common data for RLNO.py and GBNO.py")
    p.add_argument("--root", type=Path, default=Path("data"), help="Top-level data directory")
    p.add_argument("--n1d", type=int, default=256, help="Number of synthetic 1D samples")
    p.add_argument("--n2d", type=int, default=128, help="Number of synthetic 2D proxy samples")
    p.add_argument("--nx", type=int, default=64, help="Synthetic 1D grid size")
    p.add_argument("--res2d", type=int, default=32, help="Synthetic 2D proxy grid size")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--make-synthetic", action="store_true", help="Create lightweight synthetic/proxy datasets")
    p.add_argument("--download-pdebench", nargs="*", choices=list(PDEBENCH_2D.keys()), help="Download selected official PDEBench 2D datasets")
    p.add_argument("--max-pdebench-samples", type=int, default=256, help="Subset size to standardize after download")
    p.add_argument("--target-grid", type=int, default=64, help="Downsample official PDEBench spatial grid to this size")
    p.add_argument("--overwrite", action="store_true", help="Overwrite raw downloads")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    common_dir = args.root / "common"
    raw_dir = args.root / "pdebench_raw"

    # Default should be useful: no args -> make synthetic smoke-test data.
    if not args.make_synthetic and args.download_pdebench is None:
        args.make_synthetic = True

    if args.make_synthetic:
        make_synthetic_all(common_dir, args.n1d, args.n2d, args.nx, args.res2d, args.seed)

    if args.download_pdebench:
        for key in args.download_pdebench:
            info = PDEBENCH_2D[key]
            print(f"WARNING: {key} official PDEBench file is listed around {info.approx_size}.")
            raw_path = raw_dir / info.filename
            download_file(info.url, raw_path, overwrite=args.overwrite)
            standardize_pdebench_file(key, raw_path, common_dir, args.max_pdebench_samples, args.target_grid)

    print("done")


if __name__ == "__main__":
    main()
