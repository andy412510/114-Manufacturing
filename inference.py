import os
import re
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from data_utils import build_env_dict


def generate_answers(model, train_mean, train_std, env_excel_path, test_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"預測結果將儲存於：{output_folder}")

    TEST_COL_X = 'Disp. X'
    TEST_COL_Z = 'Disp. Z'

    env_dict = build_env_dict(env_excel_path)
    model.eval()

    test_files = list(Path(test_folder).glob("*.csv"))
    total_predictions = 0

    with torch.no_grad():
        for file in test_files:
            match = re.search(r'2020\d{4}', file.name)
            date_key = match.group(0) if match else None
            matched_prompt = env_dict.get(date_key, "機台運作中，無特殊環境紀錄。")

            df = pd.read_csv(file)
            if TEST_COL_X not in df.columns or TEST_COL_Z not in df.columns:
                print(f"⚠️ {file.name} 缺少欄位，已跳過。")
                continue

            df_filled = df.copy()
            df_filled[TEST_COL_X] = pd.to_numeric(df_filled[TEST_COL_X], errors='coerce')
            df_filled[TEST_COL_Z] = pd.to_numeric(df_filled[TEST_COL_Z], errors='coerce')

            missing_indices = df_filled[
                df_filled[TEST_COL_X].isna() | df_filled[TEST_COL_Z].isna()
            ].index
            file_pred_count = 0

            for idx in missing_indices:
                if idx >= 16:
                    history = df_filled.loc[idx - 16 : idx - 1, [TEST_COL_X, TEST_COL_Z]].values
                else:
                    pad_len = 16 - idx
                    pad = np.tile([train_mean[0].item(), train_mean[1].item()], (pad_len, 1))
                    if idx > 0:
                        avail = df_filled.loc[0 : idx - 1, [TEST_COL_X, TEST_COL_Z]].values
                        history = np.vstack((pad, avail))
                    else:
                        history = pad

                history[np.isnan(history[:, 0]), 0] = train_mean[0].item()
                history[np.isnan(history[:, 1]), 1] = train_mean[1].item()

                history_t = torch.tensor(history, dtype=torch.float32)
                norm_history = (history_t - train_mean) / (train_std + 1e-8)
                inputs = norm_history.unsqueeze(0).cuda()

                trend = "upward" if inputs.mean() > 0 else "downward"
                _, top_indices = torch.topk(torch.abs(inputs[0, :, 0]), k=5)
                top5_lags = top_indices.tolist()

                pred = model(inputs, trend, top5_lags, expert_knowledge=matched_prompt)

                pred_x = pred[0][0].item() * train_std[0].item() + train_mean[0].item()
                pred_z = pred[0][1].item() * train_std[1].item() + train_mean[1].item()

                df_filled.at[idx, TEST_COL_X] = round(pred_x, 6)
                df_filled.at[idx, TEST_COL_Z] = round(pred_z, 6)
                file_pred_count += 1
                total_predictions += 1

            out_path = os.path.join(output_folder, f"Answer_{file.name}")
            df_filled.to_csv(out_path, index=False)
            print(f"完成：{file.name} | 填入 {file_pred_count} 筆")

    print(f"\n總計填入 {total_predictions} 筆數據。")
