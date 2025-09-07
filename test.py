import os
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import (
    Baseline,
    NBeats,
    RecurrentNetwork,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    QuantileLoss
)
from pytorch_forecasting.metrics import MAE, SMAPE, MultiLoss, RMSE
from lightning.pytorch.tuner import Tuner
from datetime import datetime

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

def static(df):
    disp_x_stats = df['disp_x'].describe()
    disp_z_stats = df['disp_z'].describe()
    print("Disp. X 統計數據:\n", disp_x_stats)
    print("\nDisp. Z 統計數據:\n", disp_z_stats)

def clean_df(df):
    sanitized_columns = [
        col.strip().lower().replace(' ', '_').replace('.', '')
        for col in df.columns
    ]
    df.columns = sanitized_columns
    return df

def parse_temperature_info(temp_str):
    if pd.isna(temp_str):
        return np.nan, np.nan
    temp_str = str(temp_str)
    cleaned_str = re.sub(r'\[.*\]', '', temp_str)
    cleaned_str = re.sub(r'\(.*\)', '', cleaned_str)
    cleaned_str = cleaned_str.strip()
    if '→' in cleaned_str:
        parts = cleaned_str.split('→')
        try:
            start_temp = float(parts[0].strip())
            end_temp = float(parts[1].strip())
            return start_temp, end_temp
        except (ValueError, IndexError):
            return np.nan, np.nan
    else:
        try:
            temp_val = float(cleaned_str)
            return temp_val, temp_val
        except ValueError:
            return np.nan, np.nan

def load_and_preprocess_data(data_path: str, settings_filepath: str, prefix: str) -> pd.DataFrame:
    data_path = Path(data_path)
    csv_files = list(data_path.glob('*.csv'))

    try:
        settings_df = pd.read_excel(settings_filepath, skiprows=2, header=None)
        settings_columns = [
            'date', 's1_speed', 's1_feed', 's1_time',
            's2_speed', 's2_feed', 's2_time',
            's3_speed', 's3_feed', 's3_time',
            'temp_control_method', 'temp_info'
        ]
        settings_df.columns = settings_columns
        settings_df['date'] = settings_df['date'].astype(str)
        settings_df['temp_control_method'] = settings_df['temp_control_method'].astype(str)
        temp_parsed = settings_df['temp_info'].apply(parse_temperature_info)
        settings_df[['start_temp', 'end_temp']] = pd.DataFrame(temp_parsed.tolist(), index=settings_df.index)
        settings_df.set_index('date', inplace=True)
    except Exception as e:
        print(f"處理 Excel 檔案時出錯: {e}")
        return pd.DataFrame()

    df_list = []
    print("\n--- 開始掃描 CSV 檔案並檢查缺失值與無窮大值 ---")
    for file_path in csv_files:
        temp_df = pd.read_csv(file_path)
        temp_df = clean_df(temp_df)

        temp_df['disp_x_is_missing'] = (temp_df['disp_x'].isna() | np.isinf(temp_df['disp_x'])).astype(str)
        temp_df['disp_z_is_missing'] = (temp_df['disp_z'].isna() | np.isinf(temp_df['disp_z'])).astype(str)

        nan_count_x = temp_df['disp_x'].isna().sum()
        nan_count_z = temp_df['disp_z'].isna().sum()
        inf_count_x = np.isinf(temp_df['disp_x']).sum()
        inf_count_z = np.isinf(temp_df['disp_z']).sum()

        if nan_count_x > 0 or nan_count_z > 0 or inf_count_x > 0 or inf_count_z > 0:
            print(f"!!! 警告：在檔案 '{file_path.name}' 中發現問題值 !!!")
            if nan_count_x > 0:
                print(f"    -> 'disp_x' 欄位有 {nan_count_x} 個缺失值 (NaN)。")
            if nan_count_z > 0:
                print(f"    -> 'disp_z' 欄位有 {nan_count_z} 個缺失值 (NaN)。")
            if inf_count_x > 0:
                print(f"    -> 'disp_x' 欄位有 {inf_count_x} 個無窮大值 (inf)。")
            if inf_count_z > 0:
                print(f"    -> 'disp_z' 欄位有 {inf_count_z} 個無窮大值 (inf)。")

        temp_df['disp_x'] = temp_df['disp_x'].replace([np.inf, -np.inf], np.nan)
        temp_df['disp_z'] = temp_df['disp_z'].replace([np.inf, -np.inf], np.nan)
        temp_df['disp_x'] = temp_df['disp_x'].interpolate(method='linear', limit_direction='both').ffill().bfill()
        temp_df['disp_z'] = temp_df['disp_z'].interpolate(method='linear', limit_direction='both').ffill().bfill()

        if temp_df['disp_x'].isna().any() or temp_df['disp_z'].isna().any():
            print(f"錯誤：在檔案 '{file_path.name}' 中填充後仍有 NaN 值，無法繼續處理。")
            continue

        filename_stem = file_path.stem
        match = re.search(r"_(\d{8})_", filename_stem)
        if not match:
            print(f"警告: 檔案 '{filename_stem}' 名稱格式不符，無法提取日期，將跳過。")
            continue
        date_str = match.group(1)

        try:
            file_settings = settings_df.loc[date_str]
            temp_df['temp_control_method'] = file_settings['temp_control_method']
            temp_df['start_temp'] = file_settings['start_temp']
            temp_df['end_temp'] = file_settings['end_temp']

            elapsed_time = 0.0
            if pd.notna(file_settings['s1_time']):
                temp_df['轉速'] = file_settings['s1_speed']
                temp_df['進給'] = file_settings['s1_feed']
                elapsed_time += file_settings['s1_time']
            if pd.notna(file_settings['s2_time']):
                temp_df.loc[temp_df['time'] >= elapsed_time, '轉速'] = file_settings['s2_speed']
                temp_df.loc[temp_df['time'] >= elapsed_time, '進給'] = file_settings['s2_feed']
                elapsed_time += file_settings['s2_time']
            if pd.notna(file_settings['s3_time']):
                temp_df.loc[temp_df['time'] >= elapsed_time, '轉速'] = file_settings['s3_speed']
                temp_df.loc[temp_df['time'] >= elapsed_time, '進給'] = file_settings['s3_feed']
        except KeyError:
            print(f"警告: 在設定檔中找不到日期為 '{date_str}' 的設定，將跳過檔案 '{filename_stem}'。")
            continue

        temp_df['group_id'] = f"{prefix}_{filename_stem}"
        temp_df['date'] = date_str
        temp_df = temp_df.sort_values(by='time').reset_index(drop=True)
        temp_df['time_idx'] = temp_df.index
        temp_df['disp_x_diff'] = temp_df['disp_x'].diff().fillna(0)
        temp_df['disp_z_diff'] = temp_df['disp_z'].diff().fillna(0)

        df_list.append(temp_df)

    if not df_list:
        print("錯誤：沒有成功載入任何數據。")
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

def build_dataset(args, train_df, test_df, df_all):
    encoder_length = args.max_encoder_length
    prediction_length = args.max_prediction_length
    b_size = args.batch_size
    num_workers = args.num_workers
    validation_cutoff = train_df["time_idx"].max() - prediction_length
    training_df_for_dataset = train_df[lambda x: x.time_idx <= validation_cutoff]

    training_dataset = TimeSeriesDataSet(
        training_df_for_dataset,
        time_idx="time_idx",
        target=["disp_x_diff", "disp_z_diff"],
        group_ids=["group_id"],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=["group_id", "date", "temp_control_method"],
        static_reals=["start_temp", "end_temp"],
        time_varying_known_reals=["轉速", "進給"],
        time_varying_known_categoricals=[],
        time_varying_unknown_reals=[
            "disp_x", "disp_z", "disp_x_diff", "disp_z_diff",
            'pt01', 'pt02', 'pt03', 'pt04', 'pt05', 'pt06', 'pt07', 'pt08', 'pt09', 'pt10', 'pt11', 'pt12', 'pt13',
            'tc01', 'tc02', 'tc03', 'tc04', 'tc05', 'tc06', 'tc07', 'tc08',
            'spindle_motor', 'x_motor', 'z_motor'
        ],
        time_varying_unknown_categoricals=["disp_x_is_missing", "disp_z_is_missing"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        categorical_encoders={
            'group_id': NaNLabelEncoder().fit(df_all['group_id']),
            'date': NaNLabelEncoder().fit(df_all['date']),
            'temp_control_method': NaNLabelEncoder().fit(df_all['temp_control_method']),
            'disp_x_is_missing': NaNLabelEncoder(add_nan=True).fit(df_all['disp_x_is_missing']),
            'disp_z_is_missing': NaNLabelEncoder(add_nan=True).fit(df_all['disp_z_is_missing'])
        }
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, train_df, predict=True,
                                                       stop_randomization=True)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=b_size, num_workers=num_workers)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=b_size * 10, num_workers=num_workers)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df, stop_randomization=True)
    test_dataloader = test_dataset.to_dataloader(batch_size=b_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, training_dataset, test_dataloader

def training(args, train_dataloader, val_dataloader, training_dataset):
    epochs = args.epochs
    hidden_size = args.hidden_size
    attention_head_size = args.attention_head_size
    dropout = args.dropout
    hidden_continuous_size = args.hidden_continuous_size
    best_dir = args.best_dir
    model_name = args.model_name

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        optimizer="adam",
        reduce_on_plateau_patience=4,
    )

    print(f"模型中的參數數量: {tft.size() / 1e6:.2f} M")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=best_dir,
        filename=model_name,
        save_top_k=1,
        mode="min"
    )

    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback
        ]
    )

    tuner = Tuner(trainer)
    lr_find_results = tuner.lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=1.0,
        min_lr=1e-6,
    )
    tft.hparams.learning_rate = lr_find_results.suggestion()

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"最佳模型儲存路徑: {best_model_path}")

    return best_model_path

def tft(best_model_path, test_dataloader, training_dataset, output_dir):
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    print("已成功從檢查點載入最佳模型。")
    best_tft.eval()

    try:
        device = next(best_tft.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    print(f"模型已載入，並位於設備: {device}")

    reals_order = training_dataset.reals
    idx_x_abs = reals_order.index('disp_x')
    idx_z_abs = reals_order.index('disp_z')

    final_predictions_x = []
    final_predictions_z = []
    final_actuals_x = []
    final_actuals_z = []

    print("正在逐批次進行預測與還原...")
    with torch.no_grad():
        for x, y in iter(test_dataloader):
            x_on_device = {}
            for key, val in x.items():
                if isinstance(val, torch.Tensor):
                    x_on_device[key] = val.to(device)
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                    x_on_device[key] = [v.to(device) for v in val]
                else:
                    x_on_device[key] = val

            model_output = best_tft(x_on_device)
            pred_diffs_x = model_output["prediction"][0][:, :, 3]
            pred_diffs_z = model_output["prediction"][1][:, :, 3]

            start_x = x_on_device['encoder_cont'][:, -1, idx_x_abs]
            start_z = x_on_device['encoder_cont'][:, -1, idx_z_abs]
            actuals_x_batch = x_on_device['decoder_cont'][..., idx_x_abs]
            actuals_z_batch = x_on_device['decoder_cont'][..., idx_z_abs]

            num_predictions_batch = pred_diffs_x.shape[0]
            start_x_batch = start_x[:num_predictions_batch]
            start_z_batch = start_z[:num_predictions_batch]

            reconstructed_x = torch.cumsum(torch.cat([start_x_batch.unsqueeze(1), pred_diffs_x], dim=1), dim=1)[:, 1:]
            reconstructed_z = torch.cumsum(torch.cat([start_z_batch.unsqueeze(1), pred_diffs_z], dim=1), dim=1)[:, 1:]

            pred_len = reconstructed_x.shape[1]
            final_predictions_x.extend(reconstructed_x.flatten().cpu().numpy())
            final_predictions_z.extend(reconstructed_z.flatten().cpu().numpy())
            final_actuals_x.extend(actuals_x_batch[:, :pred_len].flatten().cpu().numpy())
            final_actuals_z.extend(actuals_z_batch[:, :pred_len].flatten().cpu().numpy())

    predictions_x = np.array(final_predictions_x)
    predictions_z = np.array(final_predictions_z)
    actuals_x = np.array(final_actuals_x)
    actuals_z = np.array(final_actuals_z)

    print("\n--- 定量效能評估 (一階差分還原後) ---")
    rmse_x = np.sqrt(mean_squared_error(actuals_x, predictions_x))
    print(f"Disp. X - RMSE: {rmse_x:.4f}")
    print("-" * 20)
    rmse_z = np.sqrt(mean_squared_error(actuals_z, predictions_z))
    print(f"Disp. Z - RMSE: {rmse_z:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    rmse_df = pd.DataFrame({
        'Metric': ['Disp. X - RMSE', 'Disp. Z - RMSE'],
        'Value': [rmse_x, rmse_z]
    })
    output_path = os.path.join(output_dir, f'rmse_results_{timestamp}.csv')
    rmse_df.to_csv(output_path, index=False)
    print(f"RMSE 結果已保存至: {output_path}")

    return 0

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_df = load_and_preprocess_data(args.train_data_dir, args.settings_dir, prefix='train')
    test_df = load_and_preprocess_data(args.test_data_dir, args.settings_dir, prefix='test')
    print("\n--- 訓練數據統計 ---")
    static(train_df)
    print("\n--- 測試數據統計 ---")
    static(test_df)
    print("\n--- 資料讀取成功！---")
    train_df['data_type'] = 'train'
    test_df['data_type'] = 'test'
    df_all = pd.concat([train_df, test_df], ignore_index=True)

    print("\n--- 類別變數數據類型 ---")
    print(f"group_id: {df_all['group_id'].dtype}")
    print(f"date: {df_all['date'].dtype}")
    print(f"temp_control_method: {df_all['temp_control_method'].dtype}")
    print(f"disp_x_is_missing: {df_all['disp_x_is_missing'].dtype}")
    print(f"disp_z_is_missing: {df_all['disp_z_is_missing'].dtype}")

    train_dataloader, val_dataloader, training_dataset, test_dataloader = build_dataset(args, train_df, test_df, df_all)
    print("\n--- TimeSeriesDataSet 和 DataLoader 建立成功！---")

    best_model_path = training(args, train_dataloader, val_dataloader, training_dataset)
    tft(best_model_path, test_dataloader, training_dataset, args.best_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="contrastive learning on unsupervised re-ID")
    parser.add_argument('--train_data_dir', type=str, metavar='PATH',
                        default='/Users/tule/Desktop/淡江/專題/TRAIN 0-5')
    parser.add_argument('--test_data_dir', type=str, metavar='PATH',
                        default='/Users/tule/Desktop/淡江/專題/TEST 0-5/初賽測驗用數據')
    parser.add_argument('--settings_dir', type=str, default='/Users/tule/Desktop/淡江/專題/初賽訓練用數據/檔案環境設定總表.xlsx',
                        help='Path to the manufacturing settings Excel file')
    parser.add_argument('--best_dir', type=str, metavar='PATH',
                        default='/Users/tule/Desktop/淡江/專題/lightning_logs')
    parser.add_argument('--model_name', type=str, default='tft_diff_setting')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-enl', '--max_encoder_length', type=int, default=64)
    parser.add_argument('-prl', '--max_prediction_length', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--attention_head_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_continuous_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    main()