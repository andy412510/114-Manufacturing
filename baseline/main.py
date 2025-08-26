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

def load_and_preprocess_data(data_path: str, prefix: str, is_test_set: bool = False) -> pd.DataFrame:
    data_path = Path(data_path)
    csv_files = list(data_path.glob('*.csv'))
    df_list = []
    for i, file_path in enumerate(csv_files):
        # 讀取單一 CSV 檔案
        temp_df = pd.read_csv(file_path)
        # 用檔案日期建立唯一的 'group_id'
        temp_df = clean_df(temp_df)
        filename_stem = file_path.stem # 取得檔名 (不含副檔名)
        # 使用正規表示式來安全地提取日期
        # r"_(\d{8})_" 的意思是尋找被底線包圍的 8 個數字
        temp_df['group_id'] = f"{prefix}_{filename_stem}"  # 建立一個保證唯一的 group_id (維持原狀)
        match = re.search(r"_(\d{8})_", filename_stem)
        if match:
            date_str = match.group(1)
        else:
            print(f"警告: 檔案 '{file_path.stem}' 名稱格式不符，無法提取日期。日期將設定為 'unknown'。")
            date_str = 'unknown'

        temp_df['date'] = date_str  # 新增 date 欄位

        # 根據浮點數時間 'Time' 進行排序
        # 這是建立正確 time_idx 的絕對前提
        temp_df = temp_df.sort_values(by='time').reset_index(drop=True)

        # 建立從 0 開始的整數 'time_idx'
        # 直接使用排序後的新索引
        temp_df['time_idx'] = temp_df.index
        # .diff() 會計算與前一行的差值
        # 每個 group 的第一筆資料會產生 NaN，我們用 0 填充
        temp_df['disp_x_diff'] = temp_df['disp_x'].diff().fillna(0)
        temp_df['disp_z_diff'] = temp_df['disp_z'].diff().fillna(0)
        # --- 修改結束 ---
        df_list.append(temp_df)

    return pd.concat(df_list, ignore_index=True)

def build_dataset(args, train_df, test_df, df_all):
    encoder_length = args.max_encoder_length  # 模型在做預測時，被允許回看多少筆歷史資料。
    prediction_length = args.max_prediction_length  # 你希望模型一次預測未來多長的資料。
    b_size = args.batch_size
    num_workers = args.num_workers
    # --- 定義訓練集的邊界 ---
    # 找到所有時間序列中，可以用於建立驗證集的分割點
    # 這個分割點是整個訓練資料的最後一個時間點，減去預測長度
    validation_cutoff = train_df["time_idx"].max() - prediction_length
    # 根據分割點建立訓練專用的 DataFrame
    training_df_for_dataset = train_df[lambda x: x.time_idx <= validation_cutoff]
    # --- 實例化 TimeSeriesDataSet ---
    training_dataset = TimeSeriesDataSet(
        training_df_for_dataset,  # 只傳入訓練資料的子集
        time_idx="time_idx",
        # target=["disp_x", "disp_z"],  #  多目標預測，傳入 list
        target=["disp_x_diff", "disp_z_diff"],  #  多目標預測，傳入 list
        group_ids=["group_id"],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        # 靜態特徵 (對於一個 group 永不改變)
        static_categoricals=["group_id", "date"],  # 將 date 作為靜態分類特徵
        # 動態特徵 (隨時間改變)，沒有已知的未來特徵，所以這兩項為空
        time_varying_known_reals=[],
        time_varying_known_categoricals=[],
        # 隨時間改變，但未來未知的特徵 (所有感測器讀數)
        time_varying_unknown_reals=[
            "disp_x", "disp_z",
            "disp_x_diff", "disp_z_diff",
            'pt01', 'pt02', 'pt03', 'pt04', 'pt05', 'pt06', 'pt07', 'pt08', 'pt09', 'pt10', 'pt11', 'pt12', 'pt13',
            'tc01', 'tc02', 'tc03', 'tc04', 'tc05', 'tc06', 'tc07', 'tc08',
            'spindle_motor', 'x_motor', 'z_motor'
        ],
        # 其他設定
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,  # 允許序列中有缺失的時間步

        # 傳入完整的 df_all 讓 dataset 學習所有 group 和分類值的編碼
        # 但它只會從 data_type == 'train' 的部分生成樣本
        categorical_encoders={'group_id': NaNLabelEncoder().fit(df_all['group_id']),
                              'date': NaNLabelEncoder().fit(df_all['date'])}
    )
    # --- 建立驗證集和 DataLoader ---
    # from_dataset 會自動使用 training_dataset 的設定 (例如正規化參數)
    # 它會自動從 training_cutoff 之後的資料點生成驗證樣本
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, train_df, predict=True,
                                                        stop_randomization=True)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=b_size,
                                                      num_workers=num_workers)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=b_size * 10, num_workers=num_workers)
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,  # 使用training_dataset的參數設定
        test_df,
        stop_randomization=True  # 確保資料順序不變
    )
    test_dataloader = test_dataset.to_dataloader(
        batch_size=b_size,  # 可以使用與訓練時相同的 batch size
        shuffle=False,  # 測試時絕對不能打亂順序
        num_workers=num_workers
    )
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
    # 設定早停法回呼 (callback)
    # 如果驗證集損失 (val_loss) 在 5 個 epoch 內都沒有改善，就提前停止訓練
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")

    # 設定學習率監控回呼
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=epochs,  # 最大訓練週期數
        accelerator="auto",  # 自動偵測 GPU 或 CPU
        # enable_model_summary=True, # 取消註解可以看模型結構
        gradient_clip_val=0.1,  # 梯度裁剪，防止梯度爆炸
        # limit_train_batches=50,  # 每個 epoch 只用 50 個 batch 來訓練，加速初期除錯
        callbacks=[
            early_stop_callback,
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

    # 顯示建議的學習率和繪圖
    # print(f"找到的最佳學習率: {lr_find_results.suggestion()}")
    # fig = lr_find_results.plot(show=True, suggest=True)
    # fig.show()

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

    return best_model_path,


def tft(best_model_path, test_dataloader, training_dataset):
    best_model_path = best_model_path[0]
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
    train_df = load_and_preprocess_data(args.train_data_dir, prefix='train', is_test_set=False)
    test_df = load_and_preprocess_data(args.test_data_dir, prefix='test', is_test_set=True)
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
    best_model_path = ('/home/user/114_Manufacturing/baseline/logs/best_model.ckpt',)
    # testing.tft(best_model_path, test_dataloader)
    tft(best_model_path, test_dataloader, training_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('--train_data_dir', type=str, metavar='PATH',
                        default='/home/user/Datasets/2025_BigData/train1/')
    parser.add_argument('--test_data_dir', type=str, metavar='PATH',
                        default='/home/user/Datasets/2025_BigData/test1/')
    parser.add_argument('--best_dir', type=str, metavar='PATH',  # best_model 參數要存哪
                        default='/home/user/114_Manufacturing/baseline/logs/')
    parser.add_argument('--model_name', type=str,
                        default='best_model',)
    # --- 設定基礎參數 ---
    # 這些參數會影響模型如何從時間序列中生成樣本
    # MAX_ENCODER_LENGTH: 模型回看的時間步長 (例如: 用過去 128 筆資料)
    # MAX_PREDICTION_LENGTH: 模型預測的未來時間步長 (例如: 預測未來 2 筆資料)
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


