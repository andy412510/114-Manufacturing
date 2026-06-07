import re
import pandas as pd
from pathlib import Path


def build_env_dict(env_excel_path):
    env_df = pd.read_excel(env_excel_path)
    env_df.columns = [
        '日期', '段1_轉速', '段1_進給', '段1_時間',
        '段2_轉速', '段2_進給', '段2_時間',
        '段3_轉速', '段3_進給', '段3_時間',
        '控溫模式', '溫度',
    ]
    env_df = env_df.drop(0).reset_index(drop=True)

    env_dict = {}
    for _, row in env_df.iterrows():
        if pd.isna(row['日期']):
            continue
        date_str = str(int(row['日期']))
        prompt = f"機台環境為{str(row['控溫模式'])}，設定溫度 {str(row['溫度'])} 度。"
        stages = []
        if pd.notna(row['段1_轉速']):
            stages.append(f"第一段轉速 {int(row['段1_轉速'])}rpm，進給 {int(row['段1_進給'])}")
        if pd.notna(row['段2_轉速']):
            stages.append(f"第二段轉速 {int(row['段2_轉速'])}rpm，進給 {int(row['段2_進給'])}")
        if stages:
            prompt += " 加工參數：" + "；".join(stages) + "。"
        env_dict[date_str] = prompt

    print(f"成功從 Excel 建立環境字典，共 {len(env_dict)} 種實驗設定。")
    return env_dict


def prepare_training_data(env_excel_path, train_folder, save_path):
    print("開始進行資料對齊與合併...")
    env_dict = build_env_dict(env_excel_path)

    csv_files = list(Path(train_folder).glob("*.csv"))
    print(f"找到 {len(csv_files)} 個 CSV 檔案，準備注入提示詞...")

    df_list = []
    for file in csv_files:
        match = re.search(r'2020\d{4}', file.name)
        if not match:
            print(f"警告：無法從檔名 {file.name} 找到日期，跳過。")
            continue

        date_key = match.group(0)
        matched_prompt = env_dict.get(date_key, "機台運作中，無特殊環境紀錄。")

        df = pd.read_csv(file)
        if 'disp_x_diff' in df.columns and 'disp_z_diff' in df.columns:
            df = df[['disp_x_diff', 'disp_z_diff']].copy()
            df['expert_prompt'] = matched_prompt
            df_list.append(df)

    result_df = pd.concat(df_list, ignore_index=True)
    result_df.to_csv(save_path, index=False)
    print(f"合併完成，共 {len(result_df)} 筆資料，已儲存至：{save_path}")
    return save_path
