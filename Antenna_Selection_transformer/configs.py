import os
import random

import numpy as np
import torch


def makedir(_dir):
    # 新建文件夹
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def setup_seed(seed=42):
    # 随机因子 保证在不同电脑上模型的复现性
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# 滑动窗口长度选择
WINDOW = 16

# 类别个数选择
CLASS_NUM = 785

# 根目录  获取当前执行脚本所在目录的绝对路径
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# 数据
DATA_DIR = os.path.join(ROOT_DIR, "data")
# DATA_PATH = os.path.join(DATA_DIR, "全信道矩阵1.csv")

# 模型
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SCALER_PATH = os.path.join(OUTPUTS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")
ACC_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "acc_visualization.png")
ACC_VISUALIZATION_CSV_PATH = os.path.join(OUTPUTS_DIR, "acc_visualization.csv")
LOSS_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "loss_visualization.png")
LOSS_VISUALIZATION_CSV_PATH = os.path.join(OUTPUTS_DIR, "loss_visualization.csv")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.txt")
# CONFUSION_MATRIX_PATH2 = os.path.join(OUTPUTS_DIR, "confusion_matrix2.png")

# makedir
makedir(DATA_DIR)
makedir(OUTPUTS_DIR)

# 模型覆盖
IS_COVER = True

# 随机种子 保证模型的可复现性
setup_seed(seed=42)

# 优先使用gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型相关配置
CONFIG = {
    "batch_size": 512,
    "lr": 0.0001,
    "epoch": 100,
    "min_epoch": 100,
    "patience": 0.0002,
    "patience_num": 60,
}
