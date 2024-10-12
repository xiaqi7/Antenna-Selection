
import math
import pandas as pd

import numpy as np

matrix_List = []                       # 全信道矩阵
fullCapacity_List = []                # 全信道容量
fullGain_List = []                    # 全信道增益


It = np.eye(2)
p = 50
Nt = 2

m = n = 8
length = 2000
I = np.eye(8)
#生成20000个信道样本
for i in range(0,length):
    A = math.sqrt(1.0/2) * (np.random.rand(m,n)+ 1j*np.random.rand(m,n))
    #将复数矩阵映射为实矩阵
    A = np.matrix(abs(A))

    #数据归一化
    nor1 = np.full(A.shape,np.max(A) - np.min(A))          #np.full填充函数(数组的形状，数组中填充的常数)
    A1 = A - np.full(A.shape,np.min(A))
    #divide(m,n)是m/n
    A1 = np.divide(A1,nor1)

    # def softmax(x):
    #     # 从每行中减去最大值，以防止数值不稳定性
    #     x -= np.max(x, axis=1, keepdims=True)
    #     exp_x = np.exp(x)
    #     return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    # A_array = np.asarray(A)
    # A1 = softmax(A_array)

    # A1 = A
    # # 将矩阵转换为 PyTorch 的张量
    # A1_tensor = torch.tensor(A1)
    # A1 = F.softmax(A1_tensor, dim=1)
    # # 将 PyTorch 张量转换为 NumPy 数组
    # array = A1.numpy()
    # # 将 NumPy 数组转换为矩阵对象
    # A1 = np.matrix(array)

    # matrixList是全信道矩阵
    t = np.array(A1).reshape(1, -1)  # reshape(1,-1)转化成1行,现在的A1是8*8的，你要转化成1*64
    matrix_List.append(t[0])

    # full_ChannelCapacity原全信道容量
    fullCapacity = np.log2(np.linalg.det(I + p * A1.T * A1 / n))
    fullCapacity_List.append(fullCapacity)
    # fullGain原全信道增益
    fullGain = math.sqrt(1 / 2) * np.linalg.norm(A1, ord='fro')
    fullGain_List.append(fullGain)


# 全信道矩阵
matrixData = pd.DataFrame(matrix_List)
# 全信道容量
fullCapacityData = pd.DataFrame(fullCapacity_List)
# 全信道增益
fullGainData = pd.DataFrame(fullGain_List)

matrixData.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\data\全信道矩阵3.csv')
fullCapacityData.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\data\全信道容量3.csv')

fullGainData.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\data\全信道增益3.csv')