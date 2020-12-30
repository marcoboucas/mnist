"""Data processing."""

import pandas as pd
import torch
import numpy as np
from torchvision import transforms


transformer = transforms.Compose([
    transforms.RandomRotation(10),
])


class MNISTDataset(torch.utils.data.Dataset):
    """Dataset."""

    def __init__(self, df: pd.DataFrame, transformer: transforms.Compose = None):
        """Init."""
        x_cols = [x for x in df.columns if x != "label"]
        df = df.reset_index()
        self.df_x = df[x_cols]
        self.df_y = df['label']
        self.transformer = transformer

    def __len__(self):
        return self.df_x.shape[0]

    def __getitem__(self, idx):
        """Get item."""
        y_values = self.df_y.loc[idx]
        x_values = self.df_x.loc[idx].to_numpy().astype("float")/255
        x_values = x_values.reshape((1, 28, 28))
        x_values = torch.tensor(x_values).float()
        if self.transformer and np.random.random() < 0.5:
            x_values = self.transformer(x_values)
        return x_values, y_values


if __name__ == "__main__":
    import os
    dataset = MNISTDataset(
        pd.read_csv(os.path.join('data', "train.csv")),
        transformer
    )
    dataset.__getitem__(0)
