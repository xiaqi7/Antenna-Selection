import torch
from numpy import mat
import math
import pandas as pd
import time
import numpy as np
import torch.nn.functional as F

start = time.perf_counter()

It = np.eye(2)
p= 10
Nt = 2

def Softmaxx(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # 防止数值溢出，减去最大值
    return e_x / e_x.sum(axis=0) # 对每一行进行归一化


def maxChannelCapacity(A):
    C_new = 0
    Count_new = 0
    Count = 0
    CC = []
    CC.append(0)
    # 行  i1,i2
    for i1 in range(0, 8):
        for i2 in range(i1 + 1, 8):
            # 列 j1,j2
            for j1 in range(0, 8):
                for j2 in range(j1 + 1, 8):
                    # 其实对于i1,i2来说 没必要！=   因为i1不可能等于i2，但是这个加保险。
                    if i1 != i2:
                        if j1 != j2:
                            B = mat(np.zeros((2, 2)), dtype=float)
                            # 矩阵进行 行列 赋值
                            B[0, 0] = A[i1, j1]
                            B[0, 1] = A[i1, j2]
                            B[1, 0] = A[i2, j1]
                            B[1, 1] = A[i2, j2]
                            # 根据西电学报的公式      C是B的信道容量
                            # MIMO系统中快速联合收发天线选择算法    （2*2矩阵一个一个计算比大小）
                            C = np.log2(np.linalg.det(It + p * B.T * B / Nt))
                            CC.append(C)
    CC_softmax = CC
    CC_softmax = Softmaxx(CC_softmax)
    Count_new = np.argmax(CC_softmax)   #取出最大值下标
    C_new = CC[Count_new]

                            # # 用count来标记标签
                            # Count = Count + 1
                            # if C > C_new:  # 求最大信道容量
                            #     C_new = C
                            #     Count_new = Count
    # 返回C_new是最大信道容量  Count_new是选出的信道标签
    return [C_new, Count_new]



# 组合之后的最大信道增益(行列2*2),标签
def maxChannelGain(A):
    G_new = 0.       # 用于更新最优的等效信道增益
    Count_new = 0   # 用于更新最优的标签
    Count = 0
    CG = []
    CG.append(0)
    # 行i1,i2
    for i1 in range(0, 8):
        for i2 in range(i1+1, 8):
            # 列 j1,j2
            for j1 in range(0, 8):
                for j2 in range(j1+1, 8):
                    # 其实对于i1,i2来说 没必要！=   因为i1不可能等于i2，但是这个加保险。
                    if i1 != i2:
                        if j1 != j2:
                            B = mat(np.zeros((2, 2)), dtype=float)
                            # 矩阵进行 行列 赋值
                            B[0, 0] = A[i1, j1]
                            B[0, 1] = A[i1, j2]
                            B[1, 0] = A[i2, j1]
                            B[1, 1] = A[i2, j2]
                            # 计算等效信道增益 根号a乘根号b于根号ab.
                            G = math.sqrt(1/2) * np.linalg.norm(B, ord='fro')
                            CG.append(G)
    CG_softmax = CG
    CG_softmax = Softmaxx(CG_softmax)
    Count_new = np.argmax(CG_softmax)  # 取出最大值下标
    G_new = CG[Count_new]
                            # # 用count来标记标签
                            # Count = Count + 1
                            # if G > G_new:
                            #    G_new = G
                            #    Count_new = Count
    # 返回C_new是最大信道增益(那个子矩阵的)  Count_new是选出的信道标签
    return [G_new, Count_new]




dataset = pd.read_csv(open(r'D:\qy-transformer\AntennaTransformer-gai\data\全信道矩阵(p=10 800w).csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8)

length = dataset.shape[0]


subCapacity_List = []                 # 子信道容量
subGain_List = []                     # 子信道容量
capacityLabel_List = []               # 容量标签
gainLabel_List = []                   # 增益标签

for i in range(length):
    print(i)
    # 行列组合  R = return[ , ]
    R1 = maxChannelCapacity(dataset[i])
    R2 = maxChannelGain(dataset[i])
    # 行列组合 对应最大信道容量 标签值
    subCapacity = R1[0]
    capacityLabel = R1[1]
    subGain = R2[0]
    gainLabel = R2[1]

    subCapacity_List.append(subCapacity)
    capacityLabel_List.append(capacityLabel)

    subGain_List.append(subGain)
    gainLabel_List.append(gainLabel)

    # 算时间
    if i % 1000 == 0:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(i)



# 子信道容量(穷举法最优2*2子矩阵)
subCapacityData = pd.DataFrame(subCapacity_List)
# 子信道增益(穷举法最优2*2子矩阵)
subGainData = pd.DataFrame(subGain_List)
# 选出的容量信道标签计数
capacityLabelData = pd.DataFrame(capacityLabel_List)
# 选出的增益信道标签计数
gainLabelData = pd.DataFrame(gainLabel_List)


subCapacityData.to_csv(r'D:\qy-transformer\AntennaTransformer-gai\data\子信道容量(p=10 800w).csv')

subGainData.to_csv(r'D:\qy-transformer\AntennaTransformer-gai\data\子信道增益(p=10 800w).csv')

capacityLabelData.to_csv(r'D:\qy-transformer\AntennaTransformer-gai\data\容量信道标签计数(p=10 800w).csv')
gainLabelData.to_csv(r'D:\qy-transformer\AntennaTransformer-gai\data\增益信道标签计数(p=10 800w).csv')

End = time.perf_counter()
print('信噪比  运行耗时',p, End-start)