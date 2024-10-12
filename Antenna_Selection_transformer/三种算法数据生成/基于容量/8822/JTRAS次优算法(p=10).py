import time

import numpy as np
import pandas as pd
import math

start = time.perf_counter()


#快开数据函数  得到对应行列坐标
x = pd.read_csv(r'C:\Users\10604\Desktop\AntennaTransformer\data\8822快速查询数据表.csv', header=None)
matrix = x.values
def GetLabel(i1, i2, j1, j2):
    for i in range(0, 784):
        if i1 == matrix[i, 1] and i2 == matrix[i, 2] and j1 == matrix[i, 3] and j2 == matrix[i,4]:
            return matrix[i][0]

# 首先从信道矩阵中选出范数最大的元素
def maxNormIndex(A):
    max_Gain = 0
    max_i = 0
    max_j = 0
    for i in range(0, 8):
        for j in range(0, 8):
            # Frobenius范数   每个元素的平方和 再开根号
            G = math.sqrt(A[i,j]*A[i,j])
            if G > max_Gain:
               max_Gain = G
               max_i = i
               max_j = j
    # 返回最大的行标签   列标签
    return [max_i, max_j]

#A Near-Optimal Joint Transmit and Receive Antenna Selection Algorithm for MIMO System
#Blum R S
#MIMO系统联合天线选择算法郎保才
def JTRAS_Capacity_Algorithm_8822(A1, p):
    B = np.mat(np.zeros((2, 2)), dtype=float)
    transmit = {0, 1, 2, 3, 4, 5, 6, 7}
    receive = {0, 1, 2, 3, 4, 5, 6, 7}
    #待选集合中剔除第一对发射和接收天线
    transmit.remove(maxNormIndex(A1)[0])
    receive.remove(maxNormIndex(A1)[1])
    #已选集合中增加第一对发射和接收天线
    selected_transmit = {maxNormIndex(A1)[0]}
    selected_receive = {maxNormIndex(A1)[1]}

    max_C = 0
    add_transmit  = 0
    add_receive = 0
    I2 = np.eye(2)

    for i in transmit:   #行
        for j in receive: #列
            #B[0,0]是从信道矩阵中选出范数最大的元素
            B[0,0] = A1[maxNormIndex(A1)[0],maxNormIndex(A1)[1]]
            #B[0,1]与B[0,0]有相同的行,与B[1,1]有相同的列
            B[0,1] = A1[maxNormIndex(A1)[0],j]
            #B[1,0]与B[0,0]有相同的列,与B[1,1]有相同的行
            B[1,0] = A1[i,maxNormIndex(A1)[1]]
            B[1,1] = A1[i,j]
            C = np.log2(np.linalg.det(I2 + p * B.T * B / 2))
            if C > max_C:
                max_C = C
                add_transmit = i
                add_receive = j

    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    return max_C,selected_transmit_list[0],selected_transmit_list[1],selected_receive_list[0],selected_receive_list[1]
def JTRAS_Gain_Algorithm_8822(A1):
    transmit = {0, 1, 2, 3, 4, 5, 6, 7}
    receive = {0, 1, 2, 3, 4, 5, 6, 7}
    #待选集合中剔除第一对发射和接收天线
    transmit.remove(maxNormIndex(A1)[0])
    receive.remove(maxNormIndex(A1)[1])
    #已选集合中增加第一对发射和接收天线
    selected_transmit = {maxNormIndex(A1)[0]}
    selected_receive = {maxNormIndex(A1)[1]}

    max_G = 0
    add_transmit = 0
    add_receive = 0
    for i in transmit:   #行
        for j in receive: #列
            B = np.mat(np.zeros((2, 2)), dtype=float)
            #B[0,0]是从信道矩阵中选出范数最大的元素
            B[0,0] = A1[maxNormIndex(A1)[0],maxNormIndex(A1)[1]]
            #B[0,1]与B[0,0]有相同的行,与B[1,1]有相同的列
            B[0,1] = A1[maxNormIndex(A1)[0],j]
            #B[1,0]与B[0,0]有相同的列,与B[1,1]有相同的行
            B[1,0] = A1[i,maxNormIndex(A1)[1]]
            B[1,1] = A1[i,j]
            G = math.sqrt(1 / 2) * np.linalg.norm(B, ord='fro')
            if G > max_G:
                max_G = G
                add_transmit = i
                add_receive = j

    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    return max_G,selected_transmit_list[0],selected_transmit_list[1],selected_receive_list[0],selected_receive_list[1]







#数据预处理
dataset = pd.read_csv(open(r'C:\Users\10604\Desktop\AntennaTransformer\data\全信道矩阵(p=20行列).csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8)




length = 200000
p = 10  #信噪比


subCapacity_List = []
subGain_List = []

capacityLabel_List = []
gainLabel_List = []


for i in range(0,200000):
    print(i)
    subCapacity,i1_Capacity,i2_Capacity,j1_Capacity,j2_Capacity = JTRAS_Capacity_Algorithm_8822(dataset[i],p)  #第一个返回的子信道容量
    subGain, i1_Gain, i2_Gain, j1_Gain, j2_Gain = JTRAS_Gain_Algorithm_8822(dataset[i])  #第一个返回的子信道增益
                                                                                                  

    subCapacity_List.append(subCapacity)
    subGain_List.append(subGain)
    capacityLabel_List.append(GetLabel(i1_Capacity,i2_Capacity,j1_Capacity,j2_Capacity))
    gainLabel_List.append(GetLabel(i1_Gain, i2_Gain, j1_Gain, j2_Gain))


# 子信道容量(JTRAS最优2*2子矩阵)
subCapacityData = pd.DataFrame(subCapacity_List)
# 选出的容量信道标签计数
capacityLabelData = pd.DataFrame(capacityLabel_List)

# 子信道容量(JTRAS最优2*2子矩阵)
subGainData = pd.DataFrame(subGain_List)
# 选出的容量信道标签计数
gainLabelData = pd.DataFrame(gainLabel_List)


subCapacityData.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\data\JTRAS子信道容量3(p=10行列).csv')
capacityLabelData.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\data\JTRAS容量标签3(p=10行列).csv')
subGainData.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\data\JTRAS子信道增益3(p=10行列).csv')
gainLabelData.to_csv(r'C:\Users\10604\Desktop\AntennaTransformer\data\JTRAS增益标签3(p=10行列).csv')

End = time.perf_counter()
print('运行耗时', End-start)
