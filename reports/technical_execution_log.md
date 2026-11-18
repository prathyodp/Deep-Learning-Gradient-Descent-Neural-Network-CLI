# Technical Execution Log

> Fill this with actual local execution results before submission. Do not present planned commands as completed runs.

## Environment
- Date/time:
- OS:
- Python version:
- CPU/GPU:
- TensorFlow version (if used):
- PyTorch version (if used):
- DVC version:

## Execution record

| Step | Command | Expected/actual result | Status |
|---|---|---|---|
| 1 | `pip install -r requirements.txt` | Dependencies installed | Pending |
| 2 | `python -m src.cli train --backend numpy --model linear` | Metrics JSON printed; artifacts created | Pending |
| 3 | `python -m src.cli train --backend numpy --model mlp --activation tanh` | MLP metrics printed | Pending |
| 4 | `python -m src.cli train --backend tensorflow --model linear` | TensorFlow run completes (if installed) | Pending |
| 5 | `python -m src.cli train --backend pytorch --model mlp` | PyTorch run completes (if installed) | Pending |
| 6 | `dvc repro` | DVC pipeline completes | Pending |
| 7 | `dvc metrics show` | Metrics displayed | Pending |

## Actual observations
Record the terminal output, final MSE/MAE/RMSE/R², training time, errors and fixes here.

## Reproducibility notes
- Random seed default: 42.
- Dataset: generated synthetic regression dataset unless `--data` is supplied.
- Split: deterministic train/test split (80/20 by default).
- Scaler: fit on training features only; applied to test features.
