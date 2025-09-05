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
import testing
import drawer
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)
"""
資料是.csv file，每一個csv file 代表同一機台，在不同日期、不同加工設定的測量數據。
訓練資料: 檔案數量=43，資料總筆數=29795。
測試資料: 檔案數量=13，測試資料的每個檔案，只有100筆資料。
訓練資料路徑: '/home/user/114_Manufacturing/2025_BigData/train1/'。
測試資料路徑: '/home/user/114_Manufacturing/2025_BigData/test1/'。
每個檔案，總共紀錄5-6小時的機台數據。
1st row 是資料的標頭名稱，從second row開始才是數據。
1st column 是資料收集時間，標頭名稱是 'Time'。
2nd to 14th column 是車床各個佈點的溫度，標頭名稱是 'PT01'到'PT13'。
15th to 22th column 是車床各個佈點的溫度，標頭名稱是 'TC01'到'TC08'。
23th to 25th column 是車床3個控制器的溫度，標頭名稱是 'Spindle Motor', 'X Motor', 'Z Motor'。
ground truth 是 26th to 27th column，代表X、 Z方向的熱變位量，標頭名稱是 'Disp. X', 'Disp. Z'。
"""

def static(df):
    disp_x_stats = df['disp_x'].describe()
    disp_z_stats = df['disp_z'].describe()
    print("Disp. X 統計數據:\n", disp_x_stats)
    print("\nDisp. Z 統計數據:\n", disp_z_stats)

def clean_df(df):
    # 使用 list comprehension 清理所有欄位名稱
    # 規則：移除前後空白 -> 轉為小寫 -> 將空格替換為底線 -> 移除句點
    sanitized_columns = [
        col.strip().lower().replace(' ', '_').replace('.', '')
        for col in df.columns
    ]
    df.columns = sanitized_columns
    return df


def parse_temperature_info(temp_str):
    """
    (全新版本) 解析更複雜的溫度資訊字串。
    能夠處理 '35 (info) [more_info]' 和 '15→25→15 (info)' 等格式。
    """
    # 步驟 1: 處理空值並確保輸入為字串
    if pd.isna(temp_str):
        return np.nan, np.nan
    temp_str = str(temp_str)

    # 步驟 2: (核心) 使用正規表示式移除所有附加資訊
    # 首先，移除方括號及其中的所有內容
    cleaned_str = re.sub(r'\[.*\]', '', temp_str)
    # 接著，移除圓括號及其中的所有內容
    cleaned_str = re.sub(r'\(.*\)', '', cleaned_str)

    # 步驟 3: 移除處理後可能留下的前後空白
    cleaned_str = cleaned_str.strip()

    # 步驟 4: 判斷是範圍值還是單一值
    if '→' in cleaned_str:
        # 使用 '→' 分割字串，這會處理 '15→25' 和 '15→25→15' 等情況
        parts = cleaned_str.split('→')
        try:
            # 根據您的需求，start_temp 是第一個數字，end_temp 是第二個數字
            start_temp = float(parts[0].strip())
            end_temp = float(parts[1].strip())  # 忽略 parts[2] 之後的所有內容
            return start_temp, end_temp
        except (ValueError, IndexError):
            # 如果轉換失敗或只有一個數字 (例如 '15→')，則回傳 NaN
            return np.nan, np.nan
    else:
        # 如果沒有箭頭，則視為單一值
        try:
            temp_val = float(cleaned_str)
            return temp_val, temp_val  # 起始和結束溫度相同
        except ValueError:
            return np.nan, np.nan


def load_and_preprocess_data(data_path: str, settings_filepath: str, prefix: str) -> pd.DataFrame:
    data_path = Path(data_path)
    csv_files = list(data_path.glob('*.csv'))

    # 讀取並預處理加工設定 Excel 檔案
    try:
        settings_df = pd.read_excel(settings_filepath, skiprows=2, header=None)

        settings_columns = [
            'date',
            's1_speed', 's1_feed', 's1_time',
            's2_speed', 's2_feed', 's2_time',
            's3_speed', 's3_feed', 's3_time',
            'temp_control_method',
            'temp_info'
        ]
        settings_df.columns = settings_columns

        settings_df['date'] = settings_df['date'].astype(str)

        # --- 呼叫新的解析函式 ---
        temp_parsed = settings_df['temp_info'].apply(parse_temperature_info)
        settings_df[['start_temp', 'end_temp']] = pd.DataFrame(temp_parsed.tolist(), index=settings_df.index)

        settings_df.set_index('date', inplace=True)
    except Exception as e:
        print(f"處理 Excel 檔案時出錯: {e}")
        return pd.DataFrame()

    df_list = []
    print("\n--- 開始掃描 CSV 檔案並檢查缺失值 ---")
    for file_path in csv_files:
        temp_df = pd.read_csv(file_path)
        temp_df = clean_df(temp_df)

        # ========================== 新增的除錯區塊 ==========================
        # 檢查 'disp_x' 或 'disp_z' 欄位是否包含任何 NaN 值
        if temp_df['disp_x'].isna().any() or temp_df['disp_z'].isna().any():
            nan_count_x = temp_df['disp_x'].isna().sum()
            nan_count_z = temp_df['disp_z'].isna().sum()

            # 如果發現 NaN，就印出警告訊息和檔案名稱
            print(f"!!! 警告：在檔案 '{file_path.name}' 中發現缺失值 (NA) !!!")
            if nan_count_x > 0:
                print(f"    -> 'disp_x' 欄位有 {nan_count_x} 個缺失值。")
            if nan_count_z > 0:
                print(f"    -> 'disp_z' 欄位有 {nan_count_z} 個缺失值。")
        # ========================== 除錯區塊結束 ==========================
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

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,

        categorical_encoders={
            'group_id': NaNLabelEncoder().fit(df_all['group_id']),
            'date': NaNLabelEncoder().fit(df_all['date']),
            'temp_control_method': NaNLabelEncoder().fit(df_all['temp_control_method'])
        }
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, train_df, predict=True,
                                                        stop_randomization=True)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=b_size, num_workers=num_workers)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=b_size * 10, num_workers=num_workers)

    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df, stop_randomization=True)
    test_dataloader = test_dataset.to_dataloader(batch_size=b_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, training_dataset, test_dataloader


def training(args,train_dataloader, val_dataloader, training_dataset):
    epochs = args.epochs
    hidden_size = args.hidden_size
    attention_head_size = args.attention_head_size
    dropout = args.dropout
    hidden_continuous_size = args.hidden_continuous_size
    best_dir = args.best_dir
    model_name = args.model_name
    # --- 步驟 1: 定義 TFT 模型 ---
    # 我們從 training_dataset 建立模型，讓模型自動學習資料結構
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        # --- 核心超參數 ---
        learning_rate=0.03,  # 初始學習率，稍後會被 lr_finder 自動調整
        hidden_size=hidden_size,  # 隱藏層大小
        attention_head_size=attention_head_size,
        dropout=dropout,  # Dropout 率
        hidden_continuous_size=hidden_continuous_size,  # 連續變數處理層的大小

        # --- 損失函數 ---
        # 使用 QuantileLoss，這是時間序列預測的標準做法
        # 它不僅預測中間值，還能預測分位數，提供預測區間
        loss=QuantileLoss(),

        # --- 優化器 ---
        # 使用 Adam 優化器
        optimizer="adam",

        # --- 其他設定 ---
        # 降低驗證集的 loss 計算頻率，加速驗證過程
        reduce_on_plateau_patience=4,
    )

    print(f"模型中的參數數量: {tft.size() / 1e6:.2f} M")
    # --- 步驟 2: 設定 PyTorch Lightning 訓練器 (Trainer) ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # 監控的指標
        dirpath=best_dir,  # << 關鍵：指定一個永久性的資料夾來存放模型
        filename=model_name,  # << 關鍵：給你的最佳模型一個固定的好名字
        save_top_k=1,  # 只儲存表現最好的那一個
        mode="min"  # val_loss 越小越好
    )

    # 設定學習率監控回呼
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=epochs,  # 最大訓練週期數
        accelerator="auto",  # 自動偵測 GPU 或 CPU
        # enable_model_summary=True, # 取消註解可以看模型結構
        gradient_clip_val=0.1,  # 梯度裁剪，防止梯度爆炸
        callbacks=[
            # TQDM 在各種終端機環境下的相容性是最好的
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback
        ]
    )
    # --- 步驟 3: 尋找最佳學習率 ---
    tuner = Tuner(trainer)
    # 執行學習率尋找演算法
    lr_find_results = tuner.lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=1.0,
        min_lr=1e-6,
    )
    # 將找到的最佳學習率賦值給模型
    tft.hparams.learning_rate = lr_find_results.suggestion()

    # --- 步驟 4: 開始訓練模型 ---

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"最佳模型儲存路徑: {best_model_path}")

    return best_model_path

def tft(best_model_path, test_dataloader, training_dataset):
    # best_model_path = best_model_path[0]
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
            # 轉移數據到指定設備
            x_on_device = {}
            for key, val in x.items():
                if isinstance(val, torch.Tensor):
                    # 如果值是張量，移動到設備
                    x_on_device[key] = val.to(device)
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                    # 如果值是張量列表，逐一移動
                    x_on_device[key] = [v.to(device) for v in val]
                else:
                    # 如果是其他類型 (例如 int 列表)，保持原樣
                    x_on_device[key] = val

            # 使用處理過、包含完整資訊的 x_on_device 進行預測
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

    return 0

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # step 1: 讀取指定目錄下的所有 CSV 檔案，並將它們處理、合併成一個 DataFrame。
    train_df = load_and_preprocess_data(args.train_data_dir, args.settings_dir, prefix='train')
    test_df = load_and_preprocess_data(args.test_data_dir, args.settings_dir, prefix='test')
    print("\n--- 資料讀取成功！---")
    train_df['data_type'] = 'train'
    test_df['data_type'] = 'test'
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # drawer.acf_plot(test_df)  # 畫資料acf
    # static(train_df)  # 統計資料mean, std...

    # step 2: 定義並實例化 TimeSeriesDataSet 物件。
    train_dataloader, val_dataloader, training_dataset, test_dataloader = build_dataset(args, train_df, test_df, df_all)
    print("\n--- TimeSeriesDataSet 和 DataLoader 建立成功！---")
    # testing.baseline(test_dataloader)  # 用baseline model 測試

    # step 3: 定義模型、設定訓練器，並開始訓練。
    # best_model_path = training(args,train_dataloader, val_dataloader, training_dataset)

    # step 4: 測試模型效能。
    # testing.tft(best_model_path, test_dataloader, training_dataset)
    best_model_path = '/home/user/114_Manufacturing/baseline/logs/tft_diff_setting_0.38_0.57.ckpt'
    tft(best_model_path, test_dataloader, training_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('--train_data_dir', type=str, metavar='PATH',
                        default='/home/user/Datasets/2025_BigData/train_0-5/')
    parser.add_argument('--test_data_dir', type=str, metavar='PATH',
                        default='/home/user/Datasets/2025_BigData/test1/')
    parser.add_argument('--settings_dir', type=str, default='/home/user/Datasets/2025_BigData/setting.xlsx',
                        help='Path to the manufacturing settings Excel file')
    parser.add_argument('--best_dir', type=str, metavar='PATH',  # best_model 參數要存哪
                        default='/home/user/114_Manufacturing/baseline/logs/')
    parser.add_argument('--model_name', type=str,
                        default='best_model',)
    # --- 設定基礎參數 ---
    # 這些參數會影響模型如何從時間序列中生成樣本
    # MAX_ENCODER_LENGTH: 模型回看的時間步長 (例如: 用過去 128 筆資料)
    # MAX_PREDICTION_LENGTH: 模型預測的未來時間步長 (例如: 預測未來 2 筆資料)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-enl', '--max_encoder_length', type=int, default=64)
    parser.add_argument('-prl', '--max_prediction_length', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--attention_head_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_continuous_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    main()


