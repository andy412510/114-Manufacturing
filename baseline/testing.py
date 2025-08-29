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



