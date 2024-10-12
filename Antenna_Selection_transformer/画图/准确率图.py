import matplotlib.pyplot as plt

# 定义SNR和各个模型对应的Accuracy数值
snr_values = [10, 20, 30, 40, 50]
resnet_accuracy = [0, 10, 20, 30, 100]
letnet_accuracy = [10, 20, 30, 40, 50]
vgg_accuracy = [20, 30, 40, 50, 60]
alexnet_accuracy = [30, 40, 50, 60, 70]
rnn_accuracy = [40, 50, 60, 70, 80]

# 创建一个图形和坐标轴对象
plt.figure(figsize=(10, 6))  # 设置图形大小

# 绘制折线图
plt.plot(snr_values, resnet_accuracy, label='ResNet', marker='o')
plt.plot(snr_values, letnet_accuracy, label='LeNet', marker='s')
plt.plot(snr_values, vgg_accuracy, label='VGG', marker='^')
plt.plot(snr_values, alexnet_accuracy, label='AlexNet', marker='D')
plt.plot(snr_values, rnn_accuracy, label='RNN', marker='p')

# 设置图表的标题和坐标轴标签
plt.title('Model Accuracy vs. SNR')
plt.xlabel('SNR')
plt.ylabel('Accuracy (%)')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 设置x轴的刻度
plt.xticks(snr_values)

# 显示图形
plt.show()