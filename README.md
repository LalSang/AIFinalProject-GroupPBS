# Runoff Forecasting Project

This package contains the prepared final project for improving National Water Model short-range streamflow forecasts using deep-learning post-processing.

## What is included

- `data/combined_modeling_table.csv`: merged USGS + NWM table with horizons 1-18.
- `data/combined_modeling_table_complete.csv`: cleaned table ready for deep-learning training.
- `results/original_nwm_test_metrics.csv`: baseline NWM test metrics by station and horizon.
- `results/original_nwm_test_summary.csv`: station-level baseline summary.
- `figures/`: baseline evaluation plots.
- `code/train_deep_models.py`: LSTM, GRU, and Transformer residual-correction training code.
- `Final_Report.docx`: written report draft.

## Train the deep-learning models

From inside the project folder:

```bash
pip install -r requirements.txt
cd code
python train_deep_models.py --input ../data/combined_modeling_table_complete.csv --output ../results/deep_learning --epochs 40 --batch-size 128
```

For a quick smoke test, reduce epochs:

```bash
python train_deep_models.py --input ../data/combined_modeling_table_complete.csv --output ../results/deep_learning --epochs 2 --batch-size 128
```

## Split policy

- Training: April 2021 through June 2022
- Validation: July 2022 through September 2022
- Testing: October 2022 through April 2023

This respects the instructor rule that October 2022 to April 2023 must not be used for training or hyperparameter tuning.
