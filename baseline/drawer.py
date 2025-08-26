from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

def acf_plot(df):
    # 假設 train_df 是您的 pandas DataFrame
    # 選擇其中一個 group_id 來觀察其時間序列特性
    one_series_z = df[df['group_id'] == df['group_id'].iloc[0]]['disp_x']

    # 繪製 ACF 圖，觀察前 100 個時間步的自我相關性
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(one_series_z, lags=50, ax=ax)
    plt.title("Disp. X - Autocorrelation Function (ACF)")
    plt.show()

    one_series_z = df[df['group_id'] == df['group_id'].iloc[0]]['disp_z']

    # 繪製 ACF 圖，觀察前 100 個時間步的自我相關性
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(one_series_z, lags=50, ax=ax)
    plt.title("Disp. Z - Autocorrelation Function (ACF)")
    plt.show()