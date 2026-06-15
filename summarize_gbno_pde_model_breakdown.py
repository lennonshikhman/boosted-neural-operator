#!/usr/bin/env python3
"""
Summarize full-size baseline vs boosted-tiny operator results by PDE and model family.

Default input is a CSV named like results_full_vs_boosted_tiny_10seeds*.csv in the
same directory as this script. The script writes:

  <input_stem>_pde_model_summary.csv
  <input_stem>_pde_model_summary.md

Definitions
-----------
performance_delta_pct = 100 * (full_rel_l2 - boosted_tiny_rel_l2) / full_rel_l2
    Positive means the boosted-tiny ensemble improves relative L2 error.

size_delta_pct = 100 * (boosted_tiny_total_params - full_params) / full_params
    Negative means the boosted-tiny ensemble is smaller.

size_reduction_pct = -size_delta_pct
    Positive means percent fewer parameters.

The default confidence interval is a two-sided 95% Student-t interval for the
mean over seeds within each (dataset, family) group. Bootstrap percentile CIs
are also available via --ci-method bootstrap.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "family",
    "seed",
    "full_params",
    "boosted_tiny_total_params",
    "full_rel_l2",
    "boosted_tiny_rel_l2",
}

# t critical values for two-sided 95% CI, df = n - 1.
# Used only if scipy is unavailable.
T_CRIT_975 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.5705818366147395,
    6: 2.4469118511449692,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
    11: 2.200985160082949,
    12: 2.1788128296634177,
    13: 2.1603686564610127,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.1098155778331806,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    21: 2.079613844727662,
    22: 2.0738730679040147,
    23: 2.0686576104190406,
    24: 2.0638985616280205,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.0518305164802833,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.0422724563012373,
}


def t_critical_975(df: int) -> float:
    """Return t_{0.975, df}, using scipy if available, otherwise a table/normal fallback."""
    if df <= 0:
        return float("nan")
    try:
        from scipy.stats import t  # type: ignore

        return float(t.ppf(0.975, df))
    except Exception:
        if df in T_CRIT_975:
            return T_CRIT_975[df]
        return 1.959963984540054  # normal approximation for large df


def mean_ci_t(values: Iterable[float]) -> Tuple[float, float, float]:
    """Mean and 95% Student-t CI for a one-dimensional sample."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    if n == 1:
        return mean, mean, mean
    sd = float(np.std(x, ddof=1))
    if sd == 0.0:
        return mean, mean, mean
    half_width = t_critical_975(n - 1) * sd / math.sqrt(n)
    return mean, mean - half_width, mean + half_width


def mean_ci_bootstrap(
    values: Iterable[float],
    *,
    n_boot: int = 10000,
    seed: int = 12345,
) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap 95% CI for a one-dimensional sample."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    if n == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def default_input_path(script_dir: Path) -> Path:
    candidates = sorted(script_dir.glob("results_full_vs_boosted_tiny_10seeds*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No input CSV supplied and no results_full_vs_boosted_tiny_10seeds*.csv "
            f"file found next to script: {script_dir}"
        )
    # Prefer the most recently modified matching file.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def add_deltas(df: pd.DataFrame, *, use_best_stage: bool = False) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    out = df.copy()

    if use_best_stage:
        if "best_stage_improvement_pct" not in out.columns:
            raise ValueError("--use-best-stage requires column best_stage_improvement_pct")
        out["performance_delta_pct"] = pd.to_numeric(out["best_stage_improvement_pct"], errors="coerce")
    elif "improvement_pct" in out.columns:
        out["performance_delta_pct"] = pd.to_numeric(out["improvement_pct"], errors="coerce")
    else:
        out["performance_delta_pct"] = (
            100.0
            * (pd.to_numeric(out["full_rel_l2"], errors="coerce") - pd.to_numeric(out["boosted_tiny_rel_l2"], errors="coerce"))
            / pd.to_numeric(out["full_rel_l2"], errors="coerce")
        )

    out["size_delta_pct"] = (
        100.0
        * (pd.to_numeric(out["boosted_tiny_total_params"], errors="coerce") - pd.to_numeric(out["full_params"], errors="coerce"))
        / pd.to_numeric(out["full_params"], errors="coerce")
    )
    out["size_reduction_pct"] = -out["size_delta_pct"]
    out["win"] = out["performance_delta_pct"] > 0.0
    return out


def summarize(
    df: pd.DataFrame,
    *,
    ci_method: str = "t",
    n_boot: int = 10000,
    boot_seed: int = 12345,
) -> pd.DataFrame:
    ci_fn = mean_ci_t if ci_method == "t" else None

    rows = []
    for (dataset, family), g in df.groupby(["dataset", "family"], sort=True):
        if ci_method == "t":
            perf_mean, perf_lo, perf_hi = mean_ci_t(g["performance_delta_pct"])
            size_mean, size_lo, size_hi = mean_ci_t(g["size_delta_pct"])
            red_mean, red_lo, red_hi = mean_ci_t(g["size_reduction_pct"])
        elif ci_method == "bootstrap":
            # Use a deterministic but group-specific offset so CIs are stable and not identical by accident.
            group_seed = boot_seed + abs(hash((str(dataset), str(family)))) % 1_000_000
            perf_mean, perf_lo, perf_hi = mean_ci_bootstrap(
                g["performance_delta_pct"], n_boot=n_boot, seed=group_seed
            )
            size_mean, size_lo, size_hi = mean_ci_bootstrap(
                g["size_delta_pct"], n_boot=n_boot, seed=group_seed + 1
            )
            red_mean, red_lo, red_hi = mean_ci_bootstrap(
                g["size_reduction_pct"], n_boot=n_boot, seed=group_seed + 2
            )
        else:
            raise ValueError(f"Unknown CI method: {ci_method}")

        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "n_seeds": int(g["seed"].nunique()),
                "n_rows": int(len(g)),
                "wins": int(g["win"].sum()),
                "win_rate_pct": 100.0 * float(g["win"].mean()),
                "mean_full_rel_l2": float(g["full_rel_l2"].mean()),
                "mean_boosted_tiny_rel_l2": float(g["boosted_tiny_rel_l2"].mean()),
                "full_params_mean": float(g["full_params"].mean()),
                "boosted_tiny_params_mean": float(g["boosted_tiny_total_params"].mean()),
                "mean_performance_delta_pct": perf_mean,
                "performance_delta_ci_low_pct": perf_lo,
                "performance_delta_ci_high_pct": perf_hi,
                "mean_size_delta_pct": size_mean,
                "size_delta_ci_low_pct": size_lo,
                "size_delta_ci_high_pct": size_hi,
                "mean_size_reduction_pct": red_mean,
                "size_reduction_ci_low_pct": red_lo,
                "size_reduction_ci_high_pct": red_hi,
            }
        )

    summary = pd.DataFrame(rows)
    family_order = {"fno": 0, "deeponet": 1, "cno": 2}
    summary["_family_order"] = summary["family"].map(family_order).fillna(99)
    summary = summary.sort_values(["dataset", "_family_order", "family"]).drop(columns="_family_order")
    return summary.reset_index(drop=True)


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:+.2f}%"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{fmt_pct(lo)}, {fmt_pct(hi)}]"


def fmt_int(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{int(round(x)):,}"


def write_markdown(summary: pd.DataFrame, path: Path, *, ci_method: str, use_best_stage: bool) -> None:
    lines = []
    lines.append("# Per-PDE / Model-Class GBNO Summary")
    lines.append("")
    lines.append(f"CI method: `{ci_method}`")
    lines.append(f"Performance result: `{'best_stage_improvement_pct' if use_best_stage else 'improvement_pct/final boosted_tiny_rel_l2'}`")
    lines.append("")
    lines.append("Definitions:")
    lines.append("- Performance delta = `100 * (full_rel_l2 - boosted_tiny_rel_l2) / full_rel_l2`; positive means boosted tiny is better.")
    lines.append("- Size delta = `100 * (boosted_tiny_total_params - full_params) / full_params`; negative means boosted tiny is smaller.")
    lines.append("- Size reduction = `-size_delta`; positive means fewer parameters.")
    lines.append("")

    for dataset, g in summary.groupby("dataset", sort=True):
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(
            "| Model | n | Wins | Full params | Boosted params | Mean perf. delta | 95% CI perf. | Mean size delta | 95% CI size delta | Mean size reduction |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in g.iterrows():
            lines.append(
                "| {family} | {n} | {wins}/{nrows} | {full_params} | {boosted_params} | {perf} | {perf_ci} | {size} | {size_ci} | {red} |".format(
                    family=str(r["family"]),
                    n=int(r["n_seeds"]),
                    wins=int(r["wins"]),
                    nrows=int(r["n_rows"]),
                    full_params=fmt_int(r["full_params_mean"]),
                    boosted_params=fmt_int(r["boosted_tiny_params_mean"]),
                    perf=fmt_pct(r["mean_performance_delta_pct"]),
                    perf_ci=fmt_ci(r["performance_delta_ci_low_pct"], r["performance_delta_ci_high_pct"]),
                    size=fmt_pct(r["mean_size_delta_pct"]),
                    size_ci=fmt_ci(r["size_delta_ci_low_pct"], r["size_delta_ci_high_pct"]),
                    red=fmt_pct(r["mean_size_reduction_pct"]),
                )
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize GBNO full-vs-boosted-tiny results by PDE and model family."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        help="Path to results_full_vs_boosted_tiny_10seeds*.csv. Defaults to latest matching CSV next to this script.",
    )
    parser.add_argument(
        "--ci-method",
        choices=["t", "bootstrap"],
        default="t",
        help="CI method for the mean delta. Default: t.",
    )
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap replicates if --ci-method bootstrap.")
    parser.add_argument("--boot-seed", type=int, default=12345, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--use-best-stage",
        action="store_true",
        help="Use best_stage_improvement_pct instead of final-stage improvement_pct. Not recommended for primary reporting.",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional output summary CSV path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional output Markdown path.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_csv = args.input_csv if args.input_csv is not None else default_input_path(script_dir)
    input_csv = input_csv.expanduser().resolve()

    df = pd.read_csv(input_csv)
    df = add_deltas(df, use_best_stage=args.use_best_stage)
    summary = summarize(df, ci_method=args.ci_method, n_boot=args.n_boot, boot_seed=args.boot_seed)

    stem = input_csv.stem
    out_csv = args.out_csv or input_csv.with_name(f"{stem}_pde_model_summary.csv")
    out_md = args.out_md or input_csv.with_name(f"{stem}_pde_model_summary.md")

    summary.to_csv(out_csv, index=False)
    write_markdown(summary, out_md, ci_method=args.ci_method, use_best_stage=args.use_best_stage)

    print(f"Read:  {input_csv}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print()

    display_cols = [
        "dataset",
        "family",
        "n_seeds",
        "wins",
        "mean_performance_delta_pct",
        "performance_delta_ci_low_pct",
        "performance_delta_ci_high_pct",
        "mean_size_delta_pct",
        "size_delta_ci_low_pct",
        "size_delta_ci_high_pct",
        "mean_size_reduction_pct",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
