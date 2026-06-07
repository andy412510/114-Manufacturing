import os
import math
import numpy as np
import pandas as pd
from pathlib import Path


def calculate_pseudo_rmse(original_folder, answer_folder):
    TEST_COL_X = 'Disp. X'
    TEST_COL_Z = 'Disp. Z'

    report_data = []

    for orig_file in Path(original_folder).glob("*.csv"):
        ans_path = os.path.join(answer_folder, f"Answer_{orig_file.name}")
        if not os.path.exists(ans_path):
            continue

        df_orig = pd.read_csv(orig_file)
        df_ans = pd.read_csv(ans_path)

        if TEST_COL_X not in df_orig.columns or TEST_COL_Z not in df_orig.columns:
            continue

        df_orig[TEST_COL_X] = pd.to_numeric(df_orig[TEST_COL_X], errors='coerce')
        df_orig[TEST_COL_Z] = pd.to_numeric(df_orig[TEST_COL_Z], errors='coerce')

        missing_idx = df_orig[df_orig[TEST_COL_X].isna() | df_orig[TEST_COL_Z].isna()].index
        if len(missing_idx) == 0:
            continue

        # 線性插值作為比較基準
        df_pseudo = df_orig.copy()
        df_pseudo[TEST_COL_X] = df_pseudo[TEST_COL_X].interpolate(method='linear', limit_direction='both')
        df_pseudo[TEST_COL_Z] = df_pseudo[TEST_COL_Z].interpolate(method='linear', limit_direction='both')

        rmse_x = math.sqrt(np.mean((df_pseudo.loc[missing_idx, TEST_COL_X].values - df_ans.loc[missing_idx, TEST_COL_X].values) ** 2))
        rmse_z = math.sqrt(np.mean((df_pseudo.loc[missing_idx, TEST_COL_Z].values - df_ans.loc[missing_idx, TEST_COL_Z].values) ** 2))

        name = orig_file.name if len(orig_file.name) <= 45 else orig_file.name[:42] + "..."
        report_data.append({
            "檔案名稱": name,
            "預測題數": len(missing_idx),
            "X軸 相對RMSE": round(rmse_x, 5),
            "Z軸 相對RMSE": round(rmse_z, 5),
        })

    if not report_data:
        print("找不到對應的預測檔案或缺失值，結算失敗。")
        return

    report_data.sort(key=lambda r: r['X軸 相對RMSE'])

    print("\n" + "=" * 80)
    print("模型預測 vs 線性插值基準 (Pseudo-RMSE) 評估報告")
    print("=" * 80)
    print(f"{'檔案名稱':<45} | {'預測題數':<8} | {'X軸 相對RMSE':<14} | {'Z軸 相對RMSE':<14}")
    print("-" * 80)
    for row in report_data:
        print(f"{row['檔案名稱']:<45} | {row['預測題數']:<10} | {row['X軸 相對RMSE']:<16.5f} | {row['Z軸 相對RMSE']:<14.5f}")
    print("=" * 80)
