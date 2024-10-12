import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

# from configs import WINDOW, CLASS_NUM, DATA_PATH, SCALER_PATH
from configs import WINDOW, CLASS_NUM, SCALER_PATH
from utils import save_pkl, load_pkl


def standard_scaler(values, scaler_path, mode="train"):
    # 标准差归一化
    if mode == "train":
        scaler = StandardScaler()  # 定义标准化模型
        scaler.fit(values)  # 训练
        save_pkl(scaler_path, scaler)  # 保存
    else:
        scaler = load_pkl(scaler_path)  # 加载模型
    return scaler.transform(values)  # 转换

"""
从sclarer_path加载之前训练并保存的标准差模型意味着，在训练阶段（当mode为"train"时），函数首先计算了数据集的均值和标准差，
并将这个预处理模型（即StandardScaler实例，其中包含了这些统计信息）以某种序列化形式（在这个例子中可能是pickle格式）存储到了指定的文件路径sclarer_path。

而在非训练阶段（如测试、验证或预测新数据时），函数会从sclarer_path读取之前保存的模型对象，恢复其内部的状态，包括之前训练得到的均值和标准差。
然后，利用这些已经计算好的统计参数，对新的values数据执行同样的标准化操作，确保新数据能够按照与训练数据相同的尺度进行变换，从而保持一致性。
这对于许多机器学习算法来说是非常重要的，因为它们假设输入数据具有相同的分布和尺度。
"""

# def build_X_y(data):
#     # # 构建样本集
#     # X = []  # 输入
#     # y = []  # 输出
#     # for i in tqdm(range(0, len(data) - WINDOW - CLASS_NUM + 2), desc="loading"):  # 滑动窗口
#     #     prefix = data[i : i + WINDOW - 1, :]  # 前缀
#     #     for j in range(CLASS_NUM):  # 类别
#     #         X.append(np.concatenate((prefix, data[i + WINDOW + j - 1 : i + WINDOW + j, :]), axis=0))  # 构建样本
#     #         y.append(j)  # 构建标签
#     # # df_X = pd.DataFrame(X)
#     # # df_y = pd.DataFrame(y)
#     # # df_X.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\outputs\全信道容量.csv')
#     # # df_y.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\outputs\全信道标签.csv')
#
#     X_orig = pd.read_csv(r"C:\Users\10604\Desktop\AntennaTransformer\data\全信道矩阵.csv").iloc[:, 1:]
#     X_orig = np.asarray(X_orig, np.float32)
#     dataset = X_orig.reshape(X_orig.shape[0], 8, 8, 1)
#
#     label = pd.read_csv(r"C:\Users\10604\Desktop\AntennaTransformer\data\容量信道标签计数.csv").iloc[:, 1]
#     label = np.asarray(label, np.int32)
#     label.astype(np.int32)
#     n_class = 784
#
#     # 标签转换成独热编码
#     n_sample = label.shape[0]
#     label_array = np.zeros((n_sample, n_class))  # 样本，类别
#     for i in range(n_sample):
#         label_array[i, label[i] - 1] = 1  # 非零列赋值为1
#
#
#     return dataset, label_array     #（x构建出来每个单元是data的16行数据，如0-15行对应标签为0，0-14，16行对应标签1...）


def build_X_y():
    # 加载数据集
    X_orig = pd.read_csv(r"D:\qy-transformer\AntennaTransformer-gai\data\全信道矩阵(p=10 800w).csv").iloc[:, 1:]
    X_orig = np.asarray(X_orig, np.float32)
    # dataset = X_orig.reshape(X_orig.shape[0], 8, 8, 1)
    X_data = np.expand_dims(X_orig, axis=1)

    # 加载标签
    label = pd.read_csv(r"D:\qy-transformer\AntennaTransformer-gai\data\容量信道标签计数(p=10 800w).csv").iloc[:, 1]
    label = np.asarray(label, np.int32)
    label.astype(np.int32)
    # n_class = 784

    # # 将标签转换为独热编码
    # n_sample = label.shape[0]
    # label_array = np.zeros((n_sample, n_class))  # 样本，类别
    # for i in range(n_sample):
    #     label_array[i, label[i] - 1] = 1  # 非零列赋值为1

    X = X_data
    y = label

    return X, y





# def load_data():
#     # 加载数据
#     # data = pd.read_csv(DATA_PATH, index_col=0, nrows=2000)  # 读取数据
#
#     X, y = build_X_y()  # 构建样本集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 切分训练集和验证集
#
#     print("xTrain: ", len(X_train))
#     print("xTest: ", len(X_test))
#
#     dim_0, dim_1, dim_2 = np.array(X_train).shape  # 维度
#     X_train = (
#         standard_scaler(np.array(X_train).reshape(dim_0 * dim_1, dim_2), SCALER_PATH, mode="train")
#         .reshape(dim_0, dim_1, dim_2)
#         .tolist()
#     )  # 标准化 训练集训练   （有正有负，不再是高斯？）
#
#     dim_0, dim_1, dim_2 = np.array(X_test).shape  # 维度
#     X_test = (
#         standard_scaler(np.array(X_test).reshape(dim_0 * dim_1, dim_2), SCALER_PATH, mode="test")
#         .reshape(dim_0, dim_1, dim_2)
#         .tolist()
#     )  # 标准化 验证集应用
#
#     return X_train, X_test, y_train, y_test

def build_extra_data():
    # 加载额外的训练数据
    X_extra_orig = pd.read_csv(r"D:\qy-transformer\AntennaTransformer-gai\data\j全信道矩阵(p=10 800w).csv").iloc[:, 1:]
    X_extra_orig = np.asarray(X_extra_orig, np.float32)
    X_extra_data = np.expand_dims(X_extra_orig, axis=1)

    label_extra = pd.read_csv(r"D:\qy-transformer\AntennaTransformer-gai\data\JTRAS容量标签(p=10 800w).csv").iloc[:, 1]
    label_extra = np.asarray(label_extra, np.int32)

    # 对额外数据进行相同的数据预处理（这里是标准化）
    dim_0, dim_1, dim_2 = np.array(X_extra_data).shape
    X_extra = (
        standard_scaler(np.array(X_extra_data).reshape(dim_0 * dim_1, dim_2), SCALER_PATH, mode="train")
        .reshape(dim_0, dim_1, dim_2)
        .tolist()
    )

    return X_extra, label_extra

def load_data():
    # 加载数据
    X_train, y_train = build_X_y()  # 构建训练集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.05, random_state=42)  # 切分训练集和验证集

    print("xTrain: ", len(X_train))
    print("xTest: ", len(X_test))

    dim_0, dim_1, dim_2 = np.array(X_train).shape  # 维度
    X_train = (
        standard_scaler(np.array(X_train).reshape(dim_0 * dim_1, dim_2), SCALER_PATH, mode="train")
        .reshape(dim_0, dim_1, dim_2)
        .tolist()
    )  # 标准化 训练集训练

    # 加载额外的训练数据
    X_extra, y_extra = build_extra_data()  # 假设这是一个函数，用于加载额外的训练数据，并进行了相同的数据预处理

    # 将额外数据追加到现有训练集中
    X_train = np.concatenate((X_train, X_extra), axis=0)
    y_train = np.concatenate((y_train, y_extra), axis=0)

    # 更新训练集尺寸
    print("Updated xTrain: ", len(X_train))

    dim_0, dim_1, dim_2 = np.array(X_test).shape  # 维度

    X_test_pre = X_test

    X_test = (
        standard_scaler(np.array(X_test).reshape(dim_0 * dim_1, dim_2), SCALER_PATH, mode="test")
        .reshape(dim_0, dim_1, dim_2)
        .tolist()
    )  # 标准化 验证集应用

    print("xTrain: ", len(X_train))
    print("xTest: ", len(X_test))

    # 定义权重比例
    original_weight = 1.5  # 原始数据的权重
    extra_weight = 1  # 额外数据的权重

    # 计算每个样本的权重
    num_original_samples = len(X_train) - len(X_extra)
    num_extra_samples = len(X_extra)
    weights = [original_weight] * num_original_samples + [extra_weight] * num_extra_samples

    # 创建一个WeightedRandomSampler实例
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)


    return X_train, X_test, y_train, y_test, sampler, X_test_pre



if __name__ == "__main__":
    load_data()
