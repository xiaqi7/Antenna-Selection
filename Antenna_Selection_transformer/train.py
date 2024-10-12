import os
import sys

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


from configs import (
    OUTPUTS_DIR,
    MODEL_PATH,
    ACC_VISUALIZATION_PATH,
    ACC_VISUALIZATION_CSV_PATH,
    LOSS_VISUALIZATION_PATH,
    LOSS_VISUALIZATION_CSV_PATH,
    IS_COVER,
    DEVICE,
    CONFIG,
    CONFUSION_MATRIX_PATH,
    # CONFUSION_MATRIX_PATH2,
)
from utils import save_evaluate, epoch_visualization, plot_confusion_matrix
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


def train_epoch(train_loader, model, optimizer, criterion, epoch):
    model.train()  # 训练模式
    real_targets = []  # 真实标签
    pred_targets = []  # 预测标签   （给16个，预测哪个最优？）
    train_loss_records = []  # loss
    for idx, batch_data in enumerate(tqdm(train_loader, file=sys.stdout)):  # 遍历
        inputs, targets = batch_data  # 输入 输出   （batch_data 通常是一个元组，其中包含了当前批次的数据和对应的标签）

        outputs = model(inputs.to(DEVICE))  # 前向传播

        targets = targets.reshape(-1).to(DEVICE)
        # loss = criterion(outputs, targets.reshape(-1).to(DEVICE))  # 计算loss
        loss = criterion(outputs, targets)
        optimizer.zero_grad()  # 清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        real_targets.extend(targets.reshape(-1).tolist())  # 记录真实标签
        pred_targets.extend(torch.argmax(outputs, dim=1).cpu().tolist())  # 记录预测标签
        train_loss_records.append(loss.item())  # 记录loss

    train_acc = round(accuracy_score(real_targets, pred_targets), 4)  # 计算acc
    train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)  # 求loss均值
    print(f"[train] Epoch: {epoch} / {CONFIG['epoch']}, acc: {train_acc}, loss: {train_loss}")
    return train_acc, train_loss


def evaluate(test_loader, model, criterion, epoch):
    model.eval()  # 验证模式
    real_targets = []  # 真实标签
    pred_targets = []  # 预测标签
    test_loss_records = []  # 预测标签
    for idx, batch_data in enumerate(test_loader):
        inputs, targets = batch_data  # 输入 输出

        outputs = model(inputs.to(DEVICE))  # 前向传播
        loss = criterion(outputs, targets.reshape(-1).to(DEVICE))  # 计算loss

        real_targets.extend(targets.reshape(-1).tolist())  # 记录真实标签
        pred_targets.extend(torch.argmax(outputs, dim=1).cpu().tolist())  # 记录预测标签
        test_loss_records.append(loss.item())

    test_acc = round(accuracy_score(real_targets, pred_targets), 4)  # 计算acc
    test_loss = round(sum(test_loss_records) / len(test_loss_records), 4)  # 求loss均值
    print(f"[test]  Epoch: {epoch} / {CONFIG['epoch']}, acc: {test_acc}, loss: {test_loss}")
    return test_acc, test_loss, real_targets, pred_targets


def train(train_loader, test_loader, model, optimizer, criterion):
    best_test_acc = 0  # 最佳test acc
    best_test_acc_epoch = 0
    patience_counter = 0  # 耐心值
    train_acc_records = []  # 训练acc
    train_loss_records = []  # 训练loss
    test_acc_records = []  # 测试acc
    test_loss_records = []  # 测试loss
    for epoch in range(1, CONFIG["epoch"] + 1):
        train_acc, train_loss = train_epoch(train_loader, model, optimizer, criterion, epoch)  # 训练
        test_acc, test_loss, real_targets, pred_targets = evaluate(test_loader, model, criterion, epoch)  # 验证

        train_acc_records.append(train_acc)  # 记录
        train_loss_records.append(train_loss)  # 记录
        test_acc_records.append(test_acc)  # 记录
        test_loss_records.append(test_loss)  # 记录

        if test_acc - best_test_acc > CONFIG["patience"]:
            best_test_acc = test_acc  # 记录最佳test acc
            best_test_acc_epoch = epoch
            patience_counter = 0
            torch.save(
                model.state_dict(),
                MODEL_PATH
                if IS_COVER
                else os.path.join(OUTPUTS_DIR, f"{epoch}-train_acc{train_acc}-test_acc{test_acc}-model.pkl"),
            )  # 保存模型
            # classes = [str(i) for i in range(785)]
            # plot_confusion_matrix(real_targets, pred_targets, classes, CONFUSION_MATRIX_PATH)  # 存储模型指标
            # save_evaluate(real_targets, pred_targets, CONFUSION_MATRIX_PATH, CONFUSION_MATRIX_PATH2)  # 存储模型指标
            save_evaluate(real_targets, pred_targets, CONFUSION_MATRIX_PATH)  # 存储模型指标
        else:
            patience_counter += 1

        if (patience_counter >= CONFIG["patience_num"] and epoch > CONFIG["min_epoch"] and epoch % 10 == 0) or epoch == CONFIG["epoch"]:
            print(f"best test acc: {best_test_acc}, best test acc epoch: {best_test_acc_epoch}, training finished!")
            break

    epoch_visualization(train_acc_records, test_acc_records, "accuracy", ACC_VISUALIZATION_PATH)  # 绘制acc图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_acc_records) + 1)),
            "train acc": train_acc_records,
            "test acc": test_acc_records,
        }
    ).to_csv(
        ACC_VISUALIZATION_CSV_PATH, index=False
    )  # 保存acc数据

    epoch_visualization(train_loss_records, test_loss_records, "loss", LOSS_VISUALIZATION_PATH)  # 绘制loss图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_loss_records) + 1)),
            "train loss": train_loss_records,
            "test loss": test_loss_records,
        }
    ).to_csv(
        LOSS_VISUALIZATION_CSV_PATH, index=False
    )  # 保存loss数据
