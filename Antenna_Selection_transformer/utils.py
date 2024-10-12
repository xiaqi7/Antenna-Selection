import warnings

import dill
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import seaborn as sns




from configs import CLASS_NUM

warnings.filterwarnings("ignore")  # 忽略警告
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def save_pkl(filepath, data):
    # 保存模型
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    # 加载模型
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


def save_txt(filepath, data):
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def save_evaluate(y_test, y_pred, output_path):
    report = classification_report(
        y_test,
        y_pred,
        labels=[idx for idx in range(CLASS_NUM)],
        target_names=[str(idx + 1) for idx in range(CLASS_NUM)],
        digits=4,
        zero_division=0,
    )  # 计算性能指标 包括precision/recall/f1-score/accuracy
    matrix = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
    save_txt(output_path, report + "\n\nConfusion Matrix:\n" + str(matrix))  # 保存性能指标和混淆矩阵
    # print(report + "\n\nConfusion Matrix:\n" + str(matrix))  # 输出性能指标和混淆矩阵


# def save_evaluate(y_test, y_pred, output_text_path, output_image_path):
#     report = classification_report(
#         y_test,
#         y_pred,
#         labels=[idx for idx in range(CLASS_NUM)],
#         target_names=[str(idx + 1) for idx in range(CLASS_NUM)],
#         digits=4,
#         zero_division=0,
#     )
#
#     matrix = confusion_matrix(y_test, y_pred)
#
#     # 将性能指标和混淆矩阵保存为文本文件
#     save_txt(output_text_path, report + "\n\nConfusion Matrix:\n" + str(matrix))
#
#     # 绘制混淆矩阵并保存为图片
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(matrix, cmap='Blues', interpolation='nearest')
#
#     # 设置标签
#     tick_marks = np.arange(CLASS_NUM)
#     ax.set_xticks(tick_marks)
#     ax.set_yticks(tick_marks)
#     ax.set_xticklabels([str(idx + 1) for idx in range(CLASS_NUM)])
#     ax.set_yticklabels([str(idx + 1) for idx in range(CLASS_NUM)])
#
#     # 对角线元素设置白色字体
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             text_color = 'white' if i == j else 'black'
#             ax.text(j, i, format(matrix[i, j], 'd'), ha="center", va="center", color=text_color)
#
#     ax.set_xlabel('Predicted Label')
#     ax.set_ylabel('True Label')
#     ax.set_title('Confusion Matrix')
#
#     # 保存混淆矩阵图
#     plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
#
#     # 可选：如果需要，可以在这里显示混淆矩阵图
#     # plt.show()

#
# def save_evaluate(y_test, y_pred, output_path):
#     report = classification_report(
#         y_test, y_pred,
#         labels=[idx for idx in range(CLASS_NUM)],
#         target_names=[str(idx + 1) for idx in range(CLASS_NUM)],
#         digits=4,
#         zero_division=0
#     )
#     # 计算性能指标 包括precision/recall/f1-score/accuracy
#
#     matrix = confusion_matrix(y_test, y_pred)
#     # 计算混淆矩阵
#
#     # 保存性能指标到文本文件
#     save_txt(output_path, report + "\n\nConfusion Matrix:\n" + str(matrix))
#
#     # 绘制混淆矩阵图
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(matrix, annot=True, cmap='Blues', fmt="d")
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()
#
#     # 保存混淆矩阵图为PNG文件
#     confusion_matrix_plot_path = output_path.replace('.txt', '_confusion_matrix.png')  # 假设output_path以.txt结尾
#     plt.savefig(confusion_matrix_plot_path)
#
#     # 输出性能指标（可以在控制台看到，或者重定向到其他地方）
#     print(report + "\n\nConfusion Matrix:\n" + str(matrix))
def epoch_visualization(y1, y2, name, output_path):
    plt.figure(figsize=(16, 9), dpi=100)  # 定义画布
    plt.plot(y1, marker=".", linestyle="-", linewidth=2, label=f"train {name}")  # 曲线
    plt.plot(y2, marker=".", linestyle="-", linewidth=2, label=f"test {name}")  # 曲线
    plt.title(f"训练过程中 {name} 变化图", fontsize=24)  # 标题
    plt.xlabel("epoch", fontsize=20)  # x轴标签
    plt.ylabel(name, fontsize=20)  # y轴标签
    plt.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    plt.legend(loc="best", prop={"size": 20})  # 图例
    plt.savefig(output_path)  # 保存图像
    plt.show()  # 显示




def plot_confusion_matrix(y_true, y_pred, classes, output_path, title='Confusion Matrix'):
    """
    绘制混淆矩阵的函数

    参数:
    y_true (list or array): 实际标签列表或数组
    y_pred (list or array): 预测标签列表或数组
    classes (list): 类别标签列表，用于显示在图表上的轴标签
    title (str): 图表标题，默认为'Confusion Matrix'

    返回:
    无返回值，直接绘制混淆矩阵并显示
    """

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 将混淆矩阵转换为二维数组
    cm = np.array(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

    # 添加色标条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Number of samples', rotation=-90, va="bottom")

    # 设置标签
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # 设置文字颜色（根据像素值）
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    fig.tight_layout()

    # 保存图像到指定路径
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

