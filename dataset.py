import pandas as pd
import torch
from torch.utils.data import Dataset


class RawDataset(Dataset):
    feature_columns = [
        'X_OAVelocity', 'X_Peakmg', 'X_RMSmg', 'X_Kurtosis', 'X_CrestFactor',
        'X_Skewness', 'X_Deviation', 'X_Displacement',
        'Y_OAVelocity', 'Y_Peakmg', 'Y_RMSmg', 'Y_Kurtosis', 'Y_CrestFactor',
        'Y_Skewness', 'Y_Deviation', 'Y_Displacement',
        'Z_OAVelocity', 'Z_Peakmg', 'Z_RMSmg', 'Z_Kurtosis', 'Z_CrestFactor',
        'Z_Skewness', 'Z_Deviation', 'Z_Displacement',
    ]
    conditional_columns = ['INV.HZ', 'INV.KW']

    def __init__(self, path, outlier_n_sigma=6.):
        self.outlier_n_sigma = outlier_n_sigma
        df = pd.read_csv(
            path, usecols=self.feature_columns + self.conditional_columns)
        df = df[self.feature_columns + self.conditional_columns]
        df = self.clean(df)
        self.data = df.values

    def clean(self, df):
        df = df.dropna()
        df = df.drop_duplicates()
        df = self.delete_outlier(df)
        return df

    def delete_outlier(self, df):
        is_outlier = ((df - df.mean()) / df.std()) > self.outlier_n_sigma
        is_outlier = is_outlier.any(axis="columns")
        df = df[~is_outlier]
        return df

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index])

    def __len__(self):
        return len(self.data)


class NormalizedDataset(Dataset):
    def __init__(self, dataset: Dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    @staticmethod
    def calculate_statistic(dataset: Dataset):
        data = torch.stack([x for x in dataset], dim=1)
        return torch.mean(data, dim=1), torch.std(data, dim=1)

    def __getitem__(self, index):
        x = (self.dataset[index] - self.mean) / self.std

        return (
            torch.FloatTensor(x[:len(RawDataset.feature_columns)]),
            torch.FloatTensor(x[len(RawDataset.feature_columns):]),
        )

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from torch.utils.data import Subset
    from utils import split

    dataset1 = RawDataset('./data/水泵聯軸器_20211228.csv')
    dataset2 = RawDataset('./data/水泵馬達_20211116.csv')
    pretrain_size = int(len(dataset1) * 0.5)
    # indices = list(range(len(dataset1)))
    indices = list(torch.randperm(len(dataset1)))
    pretrain_indices = indices[:pretrain_size]
    metatrain_indices = indices[pretrain_size:]

    pretrain_dataset = Subset(dataset1, pretrain_indices)
    metatrain_dataset = Subset(dataset1, metatrain_indices)
    finetune_dataset = dataset2

    # Split pretrain dataset
    pretrain_splits = split(
        pretrain_dataset, 0.2, 0.2)
    # Split metatrain dataset
    metatrain_splits = split(
        metatrain_dataset, 0.2, 0.2)
    # Split finetune dataset
    finetune_splits = split(
        finetune_dataset, 0.2, 0.6)
