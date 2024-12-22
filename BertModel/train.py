# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

from sklearn import metrics
import time
import config

import warnings
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated")

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):  # 选择数据初始化的方式
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(model, train_data_loader, dev_data_loader, test_data_loader):
    config_inf = config.Config()
    model.train()  # 设置为训练模式

    start_time = time.time()
    param_optimizer = list(model.named_parameters())  # 存放参数名
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 无需应用权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # 构建优化器和调度器
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config_inf.learning_rate)
    num_warmup_steps = int(len(train_data_loader) * config_inf.num_epochs * 0.05)
    num_training_steps = len(train_data_loader) * config_inf.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    total_batch = 0  # 记录进行到多少 batch
    dev_best_loss = float('inf')  # 保存验证集上出现过的最小损失值
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(config_inf.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config_inf.num_epochs))
        for i, (trains, labels) in enumerate(train_data_loader):
            outputs = model(trains)
            model.zero_grad()  # 模型之前积累的梯度清0
            loss = F.cross_entropy(outputs, labels)  # 交叉熵损失
            loss.backward()  # 进行反向传播，根据计算得到的损失值，自动计算模型中各个可学习参数关于该损失的梯度
            optimizer.step()  # 根据计算出的梯度使用优化器来更新模型参数
            scheduler.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()  # 找到概率最大的类别作为预测类别
                train_accuracy = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, dev_data_loader)
                if dev_loss < dev_best_loss:  # 选取损失最小作为模型保存的结果
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config_inf.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = int(time.time() - start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_accuracy, dev_loss, dev_acc, time_dif, improve))
                model.train()  # 从评估模式转回训练模式
            total_batch += 1
            if total_batch - last_improve > config_inf.require_improvement:  # 验证集loss超过1000batch没下降,结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(model, test_data_loader, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_data_loader, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    end_time = time.time()
    print("Time usage:", end_time - start_time)


def evaluate(model, data_loader, test=False):
    config_inf = config.Config()
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_loader:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        # 生成分类报告
        report = metrics.classification_report(labels_all, predict_all, target_names=config_inf.class_list, digits=4)
        # 生成混淆矩阵
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_loader), report, confusion
    return acc, loss_total / len(data_loader)
