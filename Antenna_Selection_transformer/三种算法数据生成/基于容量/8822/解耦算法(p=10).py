import time as ti

import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split

start = ti.perf_counter()


def decoupledSelection_8822(data,p):
    I = np.eye(8)
    It = np.eye(2)
    Nt = 2
    max_transmit = 0
    T1 = 0
    T2 = 0
    C = 0

    #解耦算法是发射端和接收端分开选择的
    #所以前两个for循环是要把最优的发射T1,T2选出来
    for i in range(0, 8):
        for j in range(i + 1, 8):
            B = data[[i, j], :]
            new_C = np.linalg.det(I + p * B.T * B / Nt)
            if new_C > max_transmit:
                max_transmit = new_C
                T1 = i
                T2 = j

    for i in range(0, 8):
        for j in range(i + 1, 8):
            B = data[[T1, T2]][:, [i, j]]
            new_C = np.log2(np.linalg.det(It + p * B.T * B / Nt))
            if new_C > C:
                C = new_C
    return C
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



a = 10
# 数据预处理
# iloc[:,1:].values   #从第2列到最后(因为一开始会有一个索引)
dataset = pd.read_csv(open(r"C:\Users\10604\Desktop\AntennaTransformer\data\全信道矩阵1.csv")).iloc[:, 1:]
# 变量类型转换
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)
# .iloc[:, 1]  只取第2列(因为一开始会有一个索引))
label = pd.read_csv(open(r"C:\Users\10604\Desktop\AntennaTransformer\data\容量信道标签计数1.csv")).iloc[:, 1]
# 变量类型转换
label = np.asarray(label, np.int32)
label.astype(np.int32)
# 8选2  8选2  所以就是  28 * 28
# 标签转换成独热编码
n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))  # 样本，类别
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1  # 非零列赋值为1
# 随机划分
xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
xTest_np = np.array(xTest[0:40000])


# 看公式
I1 = np.eye(8)
I2 = np.eye(2)
# 预测的容量损失列表
Pre_Loss = []
# 预测的子信道容量列表
Pre_Capacity = []
# 预测的时间列表
testTime = []
for i in range(398):
    print(i)
    # 因为存在数据集里面的时候压成1*64  转换成  8*8
    ArrayA = xTest_np[i].reshape(8, 8)
    # print(ArrayA)
    ArrayA = np.matrix(ArrayA)
    # print(ArrayA)  np.matrix不会改变数据的内部的变量类型(例如float转int)

    testStart = time()
    Pre_subCapacity = decoupledSelection_8822(ArrayA,a)
    test = time() - testStart
    # testTime里面存的是每一次预测的时间, 总时间就是最后汇总起来
    testTime.append(test)


    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)





Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)



print("基于信道容量 信噪比 ",a)
print("解耦算法_8822(200000个样本)")
print("40000个样本的测试时间 %.1f %s" % (computation_time(sum(testTime))[0], computation_time(sum(testTime))[1]))
print('预测子信道容量均值', Capacity_Mean)
print('预测损失均值', Loss_Mean)
print('预测损失方差', Loss_Variance)

End = ti.perf_counter()
print('运行耗时',End-start)
