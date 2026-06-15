
# Data



This repository does not commit large raw benchmark data files.



Expected local layout:



- `data/common/*.h5` for standardized 1D/common tasks

- external PDEBench, APEBench, and The Well data under the paths configured by `--external-data-root`



See `homogeneous_operator_boosting.py` for the exact dataset path conventions and loader globs.

