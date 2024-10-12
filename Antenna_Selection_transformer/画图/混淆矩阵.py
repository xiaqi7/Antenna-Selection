import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 假设y_true是实际标签，y_pred是模型预测的标签
# 这里仅作示例，实际需要替换为你的数据
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 2, 0, 2, 1, 0, 1, 1]

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 将混淆矩阵转换为二维数组
cm = np.array(cm)

# 绘制混淆矩阵
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()

# 设置标签
tick_marks = np.arange(len(np.unique(y_true)))
plt.xticks(tick_marks, np.unique(y_true))
plt.yticks(tick_marks, np.unique(y_true))

# 写入标签
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()