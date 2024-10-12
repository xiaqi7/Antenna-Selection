from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import CLASS_NUM, DEVICE, CONFIG
from datasets import Datasets
from models.model import Model
from models.VGG16 import VGG16
from models.Letnet import LeNet
from models.Alexnet import AlexNet
from models.Rnn import RNN
from models.Resnet18 import Resnet18
from models.Lstm import LSTM

from pretreatment import load_data
from train import train

import time

# 快开数据函数  得到对应行列坐标
x = pd.read_csv(r'D:\qy-transformer\AntennaTransformer-gai\data\8822快速查询数据表.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    # [0,784)
    if y_pre == 0:
        y_pre = 1
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            #        行1 行2  列1  列2
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]

a = 10  #信噪比


def computation_time(test):
    if test < 1e-6:
        testUnit = "ns"
        test *= 1e9
    elif test < 1e-3:
        testUnit = "us"
        test *= 1e6
    elif test < 1:
        testUnit = "ms"
        test *= 1e3
    else:
        testUnit = "s"
    return [test,testUnit]

start1 = time.perf_counter()
# def get_label_weights(labels):
#     # 标签权重
#     counts = [count for label, count in sorted(Counter(labels).most_common(), key=lambda x: x[0])]
#     label_ratios = dict(
#         zip([str(idx + 1) for idx in range(CLASS_NUM)], [round(count / sum(counts), 4) for count in counts])
#     )
#     print(f"label ratios: {label_ratios}")
#     weights = [sum(counts) / count for count in counts]
#     weights = [round(weight / sum(weights), 4) for weight in weights]
#     label_weights = dict(zip([str(idx + 1) for idx in range(CLASS_NUM)], weights))
#     print(f"label weights: {label_weights}")
#     return weights


def train_run(X_train, X_test, y_train, y_test, sampler, X_test_pre):
    train_datasets = Datasets(X_train, y_train)  # 训练数据集
    test_datasets = Datasets(X_test, y_test)  # 验证数据集

    train_loader = DataLoader(
        train_datasets,
        batch_size=CONFIG["batch_size"],
        # shuffle=True,
        num_workers=0,
        sampler=sampler,
        drop_last=False,
    )  # 训练数据加载器
    test_loader = DataLoader(
        test_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )  # 验证数据加载器

    model = Model(inputs_size=64, outputs_size=CLASS_NUM).to(DEVICE)  # 定义模型
    # model = VGG16().to(DEVICE)  # 定义模型
    # model = AlexNet().to(DEVICE)  # 定义模型
    # model = LeNet().to(DEVICE)  # 定义模型
    # model = Resnet18(785).to(DEVICE)  # 定义模型
    # model = RNN(input_size=64, output_size=785, hidden_dim=256).to(DEVICE)  # 定义模型
    # model = LSTM(input_size=64, hidden_size=256, num_layers=2, output_size=785).to(DEVICE)  # 定义模型


    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])  # 优化器
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(get_label_weights(y_train)).to(DEVICE))  # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    train(train_loader, test_loader, model, optimizer, criterion)  # 开始训练
    end1 = time.perf_counter()

    print('运行耗时', end1 - start1)

    # 将模型设置为评估模式，关闭dropout和batch normalization等训练时特有的模块行为
    model.eval()
    # 你要计算预测的时间，应该是要放在predict部分的
    # 拆成两部分预测是因为GPU的显存不够
    testStart = time.perf_counter()
    b = len(X_test_pre)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 假设 X_test 是一个NumPy数组或列表
    # 转换为PyTorch张量
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # 假设X_test中的数据是浮点数
    X_test_tensor = X_test_tensor.to(device)  # 确保输入数据也在GPU上
    # 现在 X_test_tensor 是一个PyTorch张量，你可以使用它来预测
    Model_Pre = model(X_test_tensor[0:b - 1])  # 注意这里使用圆括号而不是.predict
    testEnd = time.perf_counter()
    test = testEnd - testStart

    aaa1 = torch.argmax(Model_Pre, dim=1)
    # aaa1 = torch.argmax(Model_Pre, dim=1) + 1
    # aaa2 = np.argmax(ResNet_Pre2, axis=1) + 1
    # 首先，将张量从 GPU 转移到 CPU
    aaa1_cpu = aaa1.cpu()

    # 现在，你可以安全地将 CPU 上的张量转换为 NumPy 数组
    Model_Pre = aaa1_cpu.numpy()
    # ResNet_Pre_np2 = np.array(aaa2)
    # 将两部分预测的结果进行拼接
    # [1,2,3,4]  [5,6,7,8]
    # [1,2,3,4,5,6,7,8]
    # ResNet_Pre_np = np.hstack((ResNet_Pre_np1, ResNet_Pre_np2))
    xTest_np = np.array(X_test_pre[0:b])

    fullCapacity = pd.read_csv(r"D:\qy-transformer\AntennaTransformer-gai\data\全信道容量(p=10 800w).csv").iloc[:, 1:]
    fullCapacity = fullCapacity[0:200000]
    fullCapacity = np.asarray(fullCapacity, np.float32)

    Best_subCapacity = pd.read_csv(r"D:\qy-transformer\AntennaTransformer-gai\data\子信道容量(p=10 800w).csv").iloc[:, 1:]
    Best_subCapacity = Best_subCapacity[0:200000]
    Best_subCapacity = np.asarray(Best_subCapacity, np.float32)

    fullCapacity_Mean = np.mean(fullCapacity)
    Best_subCapacity_Mean = np.mean(Best_subCapacity)

    # 看公式
    I1 = np.eye(8)
    I2 = np.eye(2)
    # 预测的子信道损失
    Pre_Loss = []
    # 预测的子信道容量
    Pre_Capacity = []
    for i in range(b-1):
        ArrayA = xTest_np[i].reshape(8, 8)
        ArrayA = np.matrix(ArrayA)

        i1, i2, j1, j2 = GetIndexFrom(Model_Pre[i])  # 通过其反推 ij
        Pre_sub = ArrayA[[i1, i2]][:, [j1, j2]]
        Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
        Pre_subCapacity = np.log2(np.linalg.det(I2 + a * Pre_sub.T * Pre_sub / 2))

        Pre_Capacity.append(Pre_subCapacity)
        Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)

    Capacity_Mean = np.mean(Pre_Capacity)
    Loss_Mean = np.mean(Pre_Loss)
    Loss_Variance = np.var(Pre_Loss)

    print("基于信道容量 信噪比 ", a)
    # print("8822_Capacity_ResNet_OptimalLabel(200000个样本)")
    # print("160000个样本的训练时间 %.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
    print("%s个样本的测试时间 %.1f %s" % (b,computation_time(test)[0], computation_time(test)[1]))
    print("测试样本的全信道容量均值", fullCapacity_Mean)
    print("测试样本的穷举法子信道容量均值", Best_subCapacity_Mean)
    print("穷举法损失均值", fullCapacity_Mean - Best_subCapacity_Mean)
    # print('预测准确率', accuracy)
    print('预测子信道容量均值', Capacity_Mean)
    print('预测损失均值', Loss_Mean)
    print('预测损失方差', Loss_Variance)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, sampler, X_test_pre = load_data()  # 加载数据
    train_run(X_train, X_test, y_train, y_test, sampler, X_test_pre)  # 训练模型


