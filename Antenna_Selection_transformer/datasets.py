import torch
from torch.utils.data import Dataset

from pretreatment import load_data


class Datasets(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()  # 加载数据

    train_datasets = Datasets(X_train, y_train)  # 训练数据集
    test_datasets = Datasets(X_test, y_test)  # 验证数据集

    print(f"train data length: {len(train_datasets)}, test data length: {len(test_datasets)}")  # 数据集长度

    for idx, (features, targets) in enumerate(train_datasets):  # 遍历数据集
        print(features.shape, targets.shape)  # 数据维度
        print(features, targets)  # 数据示例
        break
