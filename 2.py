import os
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from pytorch_forecasting import (
    Baseline, NBeats, RecurrentNetwork, TemporalFusionTransformer, TimeSeriesDataSet,
    GroupNormalizer, MultiNormalizer, NaNLabelEncoder, QuantileLoss
)
from pytorch_forecasting.metrics import MAE, SMAPE, MultiLoss, RMSE
from lightning.pytorch.tuner import Tuner
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

def clean_df(df):
    """清理 DataFrame 欄位名稱"""
    sanitized_columns = [
        col.strip().lower().replace(' ', '_').replace('.', '')
        for col in df.columns
    ]
    df.columns = sanitized_columns
    return df

def clean_target_columns(df, target_columns=['disp_x_diff', 'disp_z_diff']):
    """清理目標欄位的缺失值和無窮大值，並新增缺失值標記"""
    for col in target_columns:
        if col not in df.columns:
            raise ValueError(f"欄位 {col} 不存在於資料集中")
        na_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        print(f"清理前 - {col}: 缺失值數量 = {na_count}, 無窮大值數量 = {inf_count}")
        
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[f'{col}_is_missing'] = df[col].isna().astype(int)
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        
        na_count_after = df[col].isna().sum()
        inf_count_after = np.isinf(df[col]).sum()
        print(f"清理後 - {col}: 缺失值數量 = {na_count_after}, 無窮大值數量 = {inf_count_after}")
    
    return df

def load_and_preprocess_data(data_path: str, prefix: str, is_test_set: bool = False) -> pd.DataFrame:
    """讀取並預處理資料，驗證差分值欄位"""
    data_path = Path(data_path)
    csv_files = list(data_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"資料目錄 {data_path} 中未找到 CSV 檔案")
    
    df_list = []
    for file_path in csv_files:
        temp_df = pd.read_csv(file_path)
        filename_stem = file_path.stem
        temp_df['group_id'] = f"{prefix}_{filename_stem}"
        match = re.search(r"_(\d{8})_", filename_stem)
        temp_df['date'] = match.group(1) if match else 'unknown'
        
        temp_df = temp_df.sort_values(by='Time').reset_index(drop=True)
        temp_df['time_idx'] = temp_df.index
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    required_columns = ['disp_x_diff', 'disp_z_diff']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"資料集中缺少欄位: {missing_cols}")
    
    return df

def build_dataset(args, train_df, test_df, df_all, target):
    """建立 TimeSeriesDataSet，支援單目標預測"""
    encoder_length = args.max_encoder_length
    prediction_length = args.max_prediction_length
    b_size = args.batch_size
    num_workers = args.num_workers
    
    validation_cutoff = train_df["time_idx"].max() - prediction_length
    training_df_for_dataset = train_df[lambda x: x.time_idx <= validation_cutoff]
    
    try:
        training_dataset = TimeSeriesDataSet(
            training_df_for_dataset,
            time_idx="time_idx",
            target=target,
            group_ids=["group_id"],
            max_encoder_length=encoder_length,
            max_prediction_length=prediction_length,
            static_categoricals=["group_id", "date"],
            time_varying_known_reals=[],
            time_varying_known_categoricals=[],
            time_varying_unknown_reals=[
                "disp_x_diff", "disp_z_diff",
                'pt01', 'pt02', 'pt03', 'pt04', 'pt05', 'pt06', 'pt07', 'pt08', 'pt09', 'pt10', 'pt11', 'pt12', 'pt13',
                'tc01', 'tc02', 'tc03', 'tc04', 'tc05', 'tc06', 'tc07', 'tc08',
                'spindle_motor', 'x_motor', 'z_motor'
            ],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            categorical_encoders={'group_id': NaNLabelEncoder().fit(df_all['group_id']),
                                'date': NaNLabelEncoder().fit(df_all['date'])}
        )
    except Exception as e:
        raise RuntimeError(f"建立 TimeSeriesDataSet 失敗 ({target}): {str(e)}")
    
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, train_df, predict=True,
                                                       stop_randomization=True)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=b_size,
                                                     num_workers=num_workers)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=b_size * 10,
                                                     num_workers=num_workers)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df, stop_randomization=True)
    test_dataloader = test_dataset.to_dataloader(batch_size=b_size, shuffle=False,
                                                num_workers=num_workers)
    
    return train_dataloader, val_dataloader, training_dataset, test_dataloader

def training(args, train_dataloader, val_dataloader, training_dataset, target):
    """訓練單目標模型"""
    epochs = args.epochs
    hidden_size = args.hidden_size
    attention_head_size = args.attention_head_size
    dropout = args.dropout
    hidden_continuous_size = args.hidden_continuous_size
    best_dir = args.best_dir
    model_name = f"best_model_{target}"
    
    target_dir = os.path.join(best_dir, target)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
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
        
        print(f"模型中的參數數量 ({target}): {tft.size() / 1e6:.2f} M")
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=target_dir,
            filename=model_name,
            save_top_k=1,
            mode="min"
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5,
                                            verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, TQDMProgressBar(refresh_rate=10), checkpoint_callback]
        )
        
        tuner = Tuner(trainer)
        lr_find_results = tuner.lr_find(tft, train_dataloaders=train_dataloader,
                                       val_dataloaders=val_dataloader,
                                       max_lr=1.0, min_lr=1e-6)
        
        tft.hparams.learning_rate = lr_find_results.suggestion()
        
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"最佳模型儲存路徑 ({target}): {best_model_path}")
        
        return best_model_path
    except Exception as e:
        print(f"訓練失敗 ({target}): {str(e)}")
        raise

def testing(best_model_path, test_dataloader, target, output_dir):
    """測試單目標模型，儲存預測結果和 RMSE 到 CSV"""
    try:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        print(f"已成功從檢查點載入最佳模型 ({target})。")
        
        predictions = best_tft.predict(test_dataloader, mode="raw")["prediction"]
        predictions = predictions.cpu().numpy()[:, :, 3]  # 選取中位數預測
        
        actuals_list = []
        print(f"正在從 DataLoader 收集真實值 ({target})...")
        for x, y in iter(test_dataloader):
            actuals_list.append(y[0])  # 單目標預測，y[0] 是目標張量
        
        actuals = torch.cat(actuals_list).cpu().numpy()
        print(f"真實值收集並拼接完成 ({target})。")
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        print(f"{target} - RMSE: {rmse:.4f}")
        
        # 儲存預測結果和真實值到 CSV
        os.makedirs(output_dir, exist_ok=True)
        predictions_df = pd.DataFrame({
            'predictions': predictions.flatten(),
            'actuals': actuals.flatten()
        })
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"預測結果已儲存至: {predictions_path}")
        
        # 儲存 RMSE 到單獨 CSV
        rmse_df = pd.DataFrame({'RMSE': [rmse]})
        rmse_path = os.path.join(output_dir, 'rmse.csv')
        rmse_df.to_csv(rmse_path, index=False)
        print(f"RMSE 已儲存至: {rmse_path}")
        
        return predictions, actuals
    except Exception as e:
        print(f"測試失敗 ({target}): {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Time series forecasting with TFT")
    parser.add_argument('--train_data_dir', type=str, default='/Users/tule/Desktop/淡江/專題/初賽訓練用數據拷貝/train')
    parser.add_argument('--test_data_dir', type=str, default='/Users/tule/Desktop/淡江/專題/初賽測驗用數據拷貝')
    parser.add_argument('--best_dir', type=str, default='/Users/tule/Desktop/淡江/專題/lightning_logs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_encoder_length', type=int, default=64)
    parser.add_argument('--max_prediction_length', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--attention_head_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_continuous_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.best_dir, exist_ok=True)
    
    try:
        train_df = load_and_preprocess_data(args.train_data_dir, prefix='train', is_test_set=False)
        test_df = load_and_preprocess_data(args.test_data_dir, prefix='test', is_test_set=True)
        print("\n--- 資料讀取成功！---")
        
        train_df['data_type'] = 'train'
        test_df['data_type'] = 'test'
        train_df = clean_df(train_df)
        test_df = clean_df(test_df)
        
        train_df = clean_target_columns(train_df, target_columns=['disp_x_diff'])
        test_df = clean_target_columns(test_df, target_columns=['disp_x_diff'])
        print("\n--- 目標欄位清理完成！---")
        
        df_all = pd.concat([train_df, test_df], ignore_index=True)
        
        targets = ['disp_x_diff']
        best_model_paths = {}
        predictions = {}
        actuals = {}
        
        for target in targets:
            print(f"\n--- 開始處理目標: {target} ---")
            train_dataloader, val_dataloader, training_dataset, test_dataloader = build_dataset(
                args, train_df, test_df, df_all, target=target
            )
            print(f"\n--- TimeSeriesDataSet 和 DataLoader 建立成功 ({target})！---")
            
            best_model_path = training(args, train_dataloader, val_dataloader, training_dataset, target=target)
            best_model_paths[target] = best_model_path
            
            output_dir = os.path.join(args.best_dir, target)
            predictions[target], actuals[target] = testing(best_model_path, test_dataloader, 
                                                         target=target, output_dir=output_dir)
        
        return best_model_paths, predictions, actuals
    
    except Exception as e:
        print(f"主程式執行失敗: {str(e)}")
        raise

if __name__ == '__main__':
    best_model_paths, predictions, actuals = main()