# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

智慧製造 (Smart Manufacturing) research project predicting CNC lathe thermal displacement (disp_x, disp_z) using time series forecasting. Two model approaches coexist:

1. **Time-LLM + ACE** — LLaMA-2-7B reprogrammed for time series, with an ACE (Active Contextual Enhancement) framework that builds prompt context from machining parameters. Designed to run on Google Colab with GPU. Code lives in `車床.ipynb` / `車床（1) .ipynb`.
2. **TFT (Temporal Fusion Transformer)** — PyTorch Lightning + `pytorch-forecasting` implementation. Entry point is `test.py`.

## Running the TFT Model

```bash
python test.py \
  --train_data_dir /path/to/TRAIN_0-5 \
  --test_data_dir  /path/to/TEST_0-5 \
  --settings_dir   /path/to/檔案環境設定總表.xlsx \
  --best_dir       /path/to/output \
  --gpu 0
```

Key hyperparameter flags: `--max_encoder_length` (default 64), `--max_prediction_length` (default 32), `--batch_size` (default 128), `--epochs` (default 100), `--hidden_size` (default 32).

Model weights download: see `README.md` for Google Drive link.

## Data Architecture

**Input CSVs** (one per machining run, filenames must contain a date string like `_20200101_`):
- `disp_x`, `disp_z` — absolute thermal displacement (target signals, X and Z axes)
- `pt01`–`pt13` — pressure/temperature sensor readings
- `tc01`–`tc08` — thermocouple readings
- `spindle_motor`, `x_motor`, `z_motor` — motor load values
- `time` — elapsed time index

**Settings Excel** (`檔案環境設定總表.xlsx`, skip first 2 rows):
Columns: date, (speed/feed/time) × 3 stages, temp_control_method, temp_info.
Date keys must match the `_YYYYMMDD_` pattern in CSV filenames.

## Prediction Target

The model predicts **first-order differences** (`disp_x_diff`, `disp_z_diff`) and then reconstructs absolute displacement via cumulative sum. This stabilises the time series for training.

## TFT Dataset Construction

`TimeSeriesDataSet` uses:
- `group_id` — one group per CSV file, prefixed with `train_` or `test_`
- Static categoricals: `group_id`, `date`, `temp_control_method`
- Static reals: `start_temp`, `end_temp`
- Time-varying known reals: `轉速` (RPM), `進給` (feed rate) — joined from the settings Excel
- Time-varying unknown reals: sensor columns + `disp_x`, `disp_z`, `disp_x_diff`, `disp_z_diff`

## Time-LLM / ACE Architecture (Notebooks)

`TimeLLMWithACE` freezes LLaMA-2-7B weights and adds three trainable components:
- `patch_embedding` — maps 2D input patches to vocabulary space (size `V_prime`)
- `W` — reprogramming matrix projecting patches into LLM hidden size
- `output_projection` — maps LLM hidden state back to 2D displacement prediction

`ACE_Playbook` builds dynamic text prompts from trend, lag analysis, and expert machining context; the last 3 strategies are kept as a rolling playbook.
