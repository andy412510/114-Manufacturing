import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from pytorch_forecasting import (
    Baseline,
    NBeats,
    RecurrentNetwork,
    TemporalFusionTransformer,
)
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

def baseline(test_dataloader):
    """
    專門用於評估 Baseline 模型效能的函式。
    """
    print("\n--- 正在評估 Baseline 模型 ---")

    # 步驟 1: 直接實例化 Baseline 模型 (它不需要訓練)
    baseline_model = Baseline()

    # 步驟 2: 執行預測
    # Baseline.predict() 會直接回傳一個 list of tensors
    # list[0] 是 disp_x 的預測, list[1] 是 disp_z 的預測
    predictions_list = baseline_model.predict(test_dataloader)

    # 將預測結果從 tensor 轉換為 numpy array
    # 注意：這裡不再需要從字典中取值
    predictions_x = predictions_list[0].cpu().numpy()
    predictions_z = predictions_list[1].cpu().numpy()

    # 步驟 3: 收集真實值 (這部分的邏輯與您原有的函式完全相同)
    actuals_x_list = []
    actuals_z_list = []
    print("正在從 DataLoader 收集真實值以供比較...")
    for x, y in iter(test_dataloader):
        actuals_x_list.append(y[0][0])
        actuals_z_list.append(y[0][1])

    actuals_x = torch.cat(actuals_x_list).cpu().numpy()
    actuals_z = torch.cat(actuals_z_list).cpu().numpy()
    print("真實值收集完成。")

    # 步驟 4: 計算並印出 RMSE (這部分的邏輯也與您原有的函式完全相同)
    print("\n--- Baseline 模型定量效能評估 ---")
    # 評估 Disp X
    rmse_x = np.sqrt(mean_squared_error(actuals_x, predictions_x))
    print(f"Disp. X - RMSE: {rmse_x:.4f}")
    print("-" * 20)
    # 評估 Disp Z
    rmse_z = np.sqrt(mean_squared_error(actuals_z, predictions_z))
    print(f"Disp. Z - RMSE: {rmse_z:.4f}")

    return 0


def tft(best_model_path, test_dataloader):
    best_model_path = best_model_path[0]
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    print("已成功從檢查點載入最佳模型。")

    # 執行預測
    raw_predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True)

    # --- 最終修正：處理堆疊在一起的多目標預測 ---

    # 1. 處理 .x 中的資料：它們是按目標分開的，所以分別拼接
    actuals_x_tensor = torch.cat(raw_predictions.x["decoder_target"][0], dim=0)
    actuals_z_tensor = torch.cat(raw_predictions.x["decoder_target"][1], dim=0)

    start_x = torch.cat(raw_predictions.x["encoder_target"][0], dim=0)[:, -1]
    start_z = torch.cat(raw_predictions.x["encoder_target"][1], dim=0)[:, -1]

    # 獲取單一目標的樣本數，這是後續拆分的關鍵
    num_samples = actuals_x_tensor.shape[0]  # 結果會是 65

    # 2. 處理 .output 中的資料：它是所有目標堆疊在一起的單一列表，先拼接成一個大張量
    # all_outputs_tensor 的形狀會是 (num_samples * 2, prediction_length, num_quantiles)，例如 (130, 32, 7)
    all_outputs_tensor = torch.cat(raw_predictions.output, dim=0)

    # 3. 根據樣本數，將堆疊的預測結果拆分回 X 和 Z
    # 前 num_samples 筆是 X 的預測，後 num_samples 筆是 Z 的預測
    output_x_tensor = all_outputs_tensor[:num_samples]
    output_z_tensor = all_outputs_tensor[num_samples:]

    # 4. 從拆分好的張量中提取差分值預測 (中位數)
    pred_diffs_x = output_x_tensor[:, :, 3]  # MEDIAN_IDX = 3
    pred_diffs_z = output_z_tensor[:, :, 3]

    # 5. 執行向量化的累加還原
    reconstructed_x = torch.cumsum(torch.cat([start_x.unsqueeze(1), pred_diffs_x], dim=1), dim=1)[:, 1:]
    reconstructed_z = torch.cumsum(torch.cat([start_z.unsqueeze(1), pred_diffs_z], dim=1), dim=1)[:, 1:]

    # --- 修正結束 ---

    # --- 將 Tensor 轉為 Numpy 並計算 RMSE ---
    predictions_x = reconstructed_x.cpu().numpy()
    predictions_z = reconstructed_z.cpu().numpy()
    actuals_x = actuals_x_tensor.cpu().numpy()
    actuals_z = actuals_z_tensor.cpu().numpy()

    print("\n--- 定量效能評估 (一階差分還原後) ---")
    rmse_x = np.sqrt(mean_squared_error(actuals_x, predictions_x))
    print(f"Disp. X - RMSE: {rmse_x:.4f}")
    print("-" * 20)
    rmse_z = np.sqrt(mean_squared_error(actuals_z, predictions_z))
    print(f"Disp. Z - RMSE: {rmse_z:.4f}")
    return 0



