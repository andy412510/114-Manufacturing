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
import xgboost as xgb
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


def calculate_custom_first_100_rmse(results_df: pd.DataFrame, max_encoder_length: int):

    print("\n--- 正在計算公式指定的綜合 RMSE (前 100 筆) ---")

    # 步驟 1: 篩選出 time_idx 在 [max_encoder_length, 99] 之間的數據點
    start_idx = max_encoder_length
    end_idx = 99
    eval_df = results_df[(results_df['time_idx'] >= start_idx) & (results_df['time_idx'] <= end_idx)].copy()

    # 步驟 2: 檢查是否有足夠的數據
    if eval_df.empty:
        print(f"分析：在您的測試數據中，沒有找到 time_idx 在 [{start_idx}, {end_idx}] 範圍內的預測點。")
        print("因此，無法計算此區間的 RMSE。")
        return

    # 步驟 3: 計算平方誤差總和
    squared_error_x = (eval_df['pred_x'] - eval_df['actual_x']) ** 2
    squared_error_z = (eval_df['pred_z'] - eval_df['actual_z']) ** 2
    sum_of_squared_errors = (squared_error_x + squared_error_z).sum()

    # 步驟 4: 根據公式計算分母 (2 * |D|)
    # |D| 是指在評估範圍內至少有一個數據點的檔案數量
    num_files = eval_df['group_id'].nunique()

    if num_files == 0:
        print("分析：沒有找到符合條件的檔案來計算 RMSE。")
        return

    denominator = 2 * num_files

    # 步驟 5: 計算最終的 RMSE
    mean_value = sum_of_squared_errors / denominator
    custom_rmse = np.sqrt(mean_value)

    print(f"公式綜合 RMSE {custom_rmse:.4f}")

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


def create_tabular_features(df, n_lags=10):
    """
    將時間序列 DataFrame 轉換為 XGBoost 適用的表格式特徵 (X) 和目標 (y)。
    """
    print(f"正在為 XGBoost 建立 {n_lags} 階的滯後特徵...")

    # 複製一份 DataFrame 以免修改原始資料
    df_tabular = df.copy()

    # 定義需要創建滯後特徵的欄位
    feature_cols = [
        'disp_x', 'disp_z', 'disp_x_diff', 'disp_z_diff',
        'pt01', 'pt02', 'pt03', 'pt04', 'pt05', 'pt06', 'pt07', 'pt08', 'pt09', 'pt10', 'pt11', 'pt12', 'pt13',
        'tc01', 'tc02', 'tc03', 'tc04', 'tc05', 'tc06', 'tc07', 'tc08',
        'spindle_motor', 'x_motor', 'z_motor', '轉速', '進給'
    ]

    # 使用 groupby 和 shift 來安全地創建滯後特徵，避免跨檔案洩漏
    for lag in range(1, n_lags + 1):
        for col in feature_cols:
            df_tabular[f'{col}_lag_{lag}'] = df_tabular.groupby('group_id')[col].shift(lag)

    # 處理靜態特徵 (讓每一行都帶有它們)
    static_cols = ['start_temp', 'end_temp', 'temp_control_method', 'date']
    for col in static_cols:
        df_tabular[col] = df_tabular[col]

    # 移除因創建滯後特徵而產生的 NaN 列
    df_tabular.dropna(inplace=True)

    # 定義最終的特徵欄位和目標欄位
    target_cols = ['disp_x_diff', 'disp_z_diff', 'disp_x', 'disp_z', 'time_idx', 'group_id']

    # X 是所有不屬於目標的欄位
    X = df_tabular.drop(columns=target_cols)
    # y 是目標欄位
    y = df_tabular[target_cols]

    # 對分類特徵進行 One-Hot 編碼
    X = pd.get_dummies(X, columns=['temp_control_method', 'date'], drop_first=True)

    return X, y


def run_xgboost_evaluation(train_df, test_df, args):
    """
    執行完整的 XGBoost 訓練、預測和評估流程。
    """
    print("\n--- 正在執行 XGBoost 模型評估 ---")

    # 步驟 1: 建立表格式數據
    # 使用與 TFT encoder 相同的長度作為參考來創建滯後特徵
    X_train, y_train = create_tabular_features(train_df, n_lags=args.max_encoder_length)
    X_test, y_test = create_tabular_features(test_df, n_lags=args.max_encoder_length)

    # 確保訓練集和測試集的欄位一致 (因為 one-hot 編碼可能產生不同欄位)
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0
    X_test = X_test[train_cols]

    # --- 訓練 X 軸模型 ---
    print("\n正在訓練 XGBoost 模型 (disp_x)...")
    xgb_x = xgb.XGBRegressor(n_estimators=100, early_stopping_rounds=10, random_state=42)
    xgb_x.fit(X_train, y_train['disp_x_diff'],
              eval_set=[(X_test, y_test['disp_x_diff'])], verbose=False)

    # --- 訓練 Z 軸模型 ---
    print("正在訓練 XGBoost 模型 (disp_z)...")
    xgb_z = xgb.XGBRegressor(n_estimators=100, early_stopping_rounds=10, random_state=42)
    xgb_z.fit(X_train, y_train['disp_z_diff'],
              eval_set=[(X_test, y_test['disp_z_diff'])], verbose=False)

    # --- 進行預測 ---
    print("\n正在進行預測...")
    pred_diff_x = xgb_x.predict(X_test)
    pred_diff_z = xgb_z.predict(X_test)

    # --- 還原預測值 ---
    # 還原的起始點是 t-1 時刻的真實絕對值，它在 X_test 中是一個滯後特徵
    start_x = X_test[f'disp_x_lag_1']
    start_z = X_test[f'disp_z_lag_1']
    pred_x = start_x + pred_diff_x
    pred_z = start_z + pred_diff_z

    # --- 建立 results_df 以便計算 RMSE ---
    results_df = pd.DataFrame({
        'group_id': y_test['group_id'],
        'time_idx': y_test['time_idx'],
        'actual_x': y_test['disp_x'],
        'pred_x': pred_x,
        'actual_z': y_test['disp_z'],
        'pred_z': pred_z,
    })

    # --- 計算並輸出 RMSE ---
    print("\n--- XGBoost 模型標準 RMSE 評估 ---")
    rmse_x = np.sqrt(mean_squared_error(results_df['actual_x'], results_df['pred_x']))
    print(f"XGBoost - Disp. X - RMSE: {rmse_x:.4f}")
    print("-" * 20)
    rmse_z = np.sqrt(mean_squared_error(results_df['actual_z'], results_df['pred_z']))
    print(f"XGBoost - Disp. Z - RMSE: {rmse_z:.4f}")

    # 計算綜合 RMSE
    calculate_custom_first_100_rmse(results_df, args.max_encoder_length)

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
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)  #
    print("已成功從檢查點載入最佳模型。")
    best_tft.eval()  #

    try:
        device = next(best_tft.parameters()).device  #
    except StopIteration:
        device = torch.device("cpu")  #
    print(f"模型已載入，並位於設備: {device}")

    reals_order = training_dataset.reals  #
    idx_x_abs = reals_order.index('disp_x')  #
    idx_z_abs = reals_order.index('disp_z')  #
    group_id_encoder = training_dataset.categorical_encoders['group_id']  #

    results_list = []
    print("正在逐批次進行預測與還原...")
    with torch.no_grad():  #
        for x, y in iter(test_dataloader):
            encoded_group_ids_tensor = x['groups']  #
            encoded_group_ids_batch = encoded_group_ids_tensor.squeeze().cpu()
            group_ids_batch = group_id_encoder.inverse_transform(encoded_group_ids_batch)

            decoder_time_idx_batch = x['decoder_time_idx']
            x_on_device = {}
            for key, val in x.items():
                if isinstance(val, torch.Tensor):
                    x_on_device[key] = val.to(device)  #
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                    x_on_device[key] = [v.to(device) for v in val]  #
                else:
                    x_on_device[key] = val

            model_output = best_tft(x_on_device)  #
            pred_diffs_x = model_output["prediction"][0][:, :, 3]  #
            pred_diffs_z = model_output["prediction"][1][:, :, 3]  #

            start_x = x_on_device['encoder_cont'][:, -1, idx_x_abs]  #
            start_z = x_on_device['encoder_cont'][:, -1, idx_z_abs]  #
            actuals_x_batch = x_on_device['decoder_cont'][..., idx_x_abs]  #
            actuals_z_batch = x_on_device['decoder_cont'][..., idx_z_abs]  #

            reconstructed_x = torch.cumsum(torch.cat([start_x.unsqueeze(1), pred_diffs_x], dim=1), dim=1)[:, 1:]  #
            reconstructed_z = torch.cumsum(torch.cat([start_z.unsqueeze(1), pred_diffs_z], dim=1), dim=1)[:, 1:]  #

            pred_len = reconstructed_x.shape[1]

            for i in range(reconstructed_x.shape[0]):
                for j in range(pred_len):
                    results_list.append({
                        'group_id': group_ids_batch[i],
                        'time_idx': decoder_time_idx_batch[i, j].item(),
                        'actual_x': actuals_x_batch[i, j].item(),
                        'pred_x': reconstructed_x[i, j].item(),
                        'actual_z': actuals_z_batch[i, j].item(),
                        'pred_z': reconstructed_z[i, j].item(),
                    })

    results_df = pd.DataFrame(results_list)

    # --- (保留部分) 計算並輸出您原有的、分離的 RMSE ---
    print("\n--- 標準 RMSE 評估 (所有可預測的點) ---")
    rmse_x = np.sqrt(mean_squared_error(results_df['actual_x'], results_df['pred_x']))  #
    print(f"Disp. X - RMSE: {rmse_x:.4f}")
    print("-" * 20)
    rmse_z = np.sqrt(mean_squared_error(results_df['actual_z'], results_df['pred_z']))  #
    print(f"Disp. Z - RMSE: {rmse_z:.4f}")

    max_encoder_length = training_dataset.max_encoder_length  #
    calculate_custom_first_100_rmse(results_df, max_encoder_length)

    # --- 原有的儲存邏輯 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")  #
    rmse_df = pd.DataFrame({
        'Metric': ['Disp. X - RMSE', 'Disp. Z - RMSE'],
        'Value': [rmse_x, rmse_z]
    })  #
    output_path = os.path.join(output_dir, f'rmse_results_{timestamp}.csv')  #
    rmse_df.to_csv(output_path, index=False)  #
    print(f"\n標準 RMSE 結果已保存至: {output_path}")

    return 0

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_df = load_and_preprocess_data(args.train_data_dir, args.settings_dir, prefix='train')
    test_df = load_and_preprocess_data(args.test_data_dir, args.settings_dir, prefix='test')

    print("\n--- 資料讀取成功！---")
    train_df['data_type'] = 'train'
    test_df['data_type'] = 'test'
    df_all = pd.concat([train_df, test_df], ignore_index=True)

    run_xgboost_evaluation(train_df, test_df, args)

    print("\n--- 開始執行 TemporalFusionTransformer 流程 ---")
    train_dataloader, val_dataloader, training_dataset, test_dataloader = build_dataset(args, train_df, test_df,
                                                                                        df_all)  #
    print("\n--- TimeSeriesDataSet 和 DataLoader 建立成功！---")  #

    # 執行 TFT 訓練
    best_model_path = training(args, train_dataloader, val_dataloader, training_dataset)  #

    # 執行 TFT 測試
    if best_model_path:
        tft(best_model_path, test_dataloader, training_dataset, args.best_dir)  #
    else:
        print("未找到最佳模型路徑，跳過 TFT 測試。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="contrastive learning on unsupervised re-ID")
    parser.add_argument('--train_data_dir', type=str, metavar='PATH',
                        default='/home/user/Datasets/2025_BigData/train_0-5_student/')
    parser.add_argument('--test_data_dir', type=str, metavar='PATH',
                        default='/home/user/Datasets/2025_BigData/test_0-5_student/')
    parser.add_argument('--settings_dir', type=str, default='/home/user/Datasets/2025_BigData/setting.xlsx',
                        help='Path to the manufacturing settings Excel file')
    parser.add_argument('--best_dir', type=str, metavar='PATH',
                        default='/home/user/114_Manufacturing/baseline/logs/')
    parser.add_argument('--model_name', type=str, default='tft_diff_student')
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