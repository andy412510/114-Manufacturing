import torch
import pandas as pd
from torch.utils.data import Dataset


class LatheDatasetWithPrompt(Dataset):
    def __init__(self, csv_path, train_mean=None, train_std=None):
        df = pd.read_csv(csv_path).dropna()
        self.raw_data = torch.tensor(df[['disp_x_diff', 'disp_z_diff']].values, dtype=torch.float32)

        if train_mean is None or train_std is None:
            self.mean = self.raw_data.mean(dim=0)
            self.std = self.raw_data.std(dim=0)
        else:
            self.mean = train_mean
            self.std = train_std

        self.data = (self.raw_data - self.mean) / (self.std + 1e-8)
        self.prompts = df['expert_prompt'].values.tolist()
        self.seq_len = 16

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_len]
        prompt = self.prompts[idx + self.seq_len - 1]
        return seq, prompt
