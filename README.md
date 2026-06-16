
# Boosted Neural Operator



Code and results for **Operator Boosting Produces Pareto-Efficient PDE Surrogates**.



This repository contains the homogeneous same-family Operator Boosting experiment driver, result summaries, and paper figures.



## Main files



- `homogeneous_operator_boosting.py` — main experiment driver

- `summarize_gbno_pde_model_breakdown.py` — summary table generator

- `results/` — completed homogeneous result CSV/JSON and per-PDE/model summary tables

- `paper/` — LaTeX source and TikZ figures

- `data/README.md` — expected local data layout



## Install



```bash

pip install -r requirements.txt

````



## Run main experiment



```bash

python homogeneous_operator_boosting.py \

  --datasets 1d_advection 1d_burgers 1d_reacdiff pdebench_darcy pdebench_2d_reacdiff ns2d shallow2d active2d cns3d mhd3d \

  --families fno deeponet cno \

  --seeds 11 22 33 44 55 66 77 88 99 111 \

  --results results/results_full_vs_boosted_tiny_10seeds.json \

  --csv results/results_full_vs_boosted_tiny_10seeds.csv

```



## Summarize results



```bash

python summarize_gbno_pde_model_breakdown.py \

  results/results_full_vs_boosted_tiny_10seeds.csv \

  --out-csv results/results_full_vs_boosted_tiny_10seeds_pde_model_summary.csv \

  --out-md results/results_full_vs_boosted_tiny_10seeds_pde_model_summary.md

```



## Data

Large benchmark files are not committed. See `data/README.md`.
