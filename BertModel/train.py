# coding: UTF-8
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import torchmetrics
from sklearn import metrics
import time
import json
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


def train(model, train_data_loader, dev_data_loader):
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
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()

    total_batch = 0  # 记录进行到多少 batch
    dev_best_loss = float('inf')  # 保存验证集上出现过的最小损失值
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(config_inf.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config_inf.num_epochs))
        for i, (trains, labels, texts) in enumerate(train_data_loader):
            outputs = model(trains)
            model.zero_grad()  # 模型之前积累的梯度清0
            loss = criterion(outputs, labels)
            loss.backward()  # 进行反向传播，根据计算得到的损失值，自动计算模型中各个可学习参数关于该损失的梯度
            optimizer.step()  # 根据计算出的梯度使用优化器来更新模型参数
            scheduler.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = outputs.data.cpu()
                print("true", true)
                print("predic", predic)  
                train_accuracy = metrics.accuracy_score(true, predic, average='macro')
                print("train_accuracy", train_accuracy)

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


def convert_numpy_int_to_python_int(lst):
    return [int(x) if isinstance(x, np.int64) else x for x in lst]

def evaluate(model, data_loader, test=False):
    config_inf = config.Config()
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    texts_all = np.array([], dtype=str)
    input_ids_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    attention_mask_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    segment_ids_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    outputs_all = np.array([[]], dtype=np.float64).reshape(0, config_inf.num_classes)
    with torch.no_grad():
        for inputs, labels, texts in data_loader:
            texts_all = np.append(texts_all, texts)
            input_ids_all = np.append(input_ids_all, inputs[0].cpu().numpy(), axis = 0)
            segment_ids_all = np.append(segment_ids_all, inputs[1].cpu().numpy(), axis = 0)
            attention_mask_all = np.append(attention_mask_all, inputs[2].cpu().numpy(), axis = 0)
            outputs = model(inputs)
            outputs_all = np.append(outputs_all, outputs.cpu().numpy(), axis = 0)

            loss = criterion(outputs, labels)
            loss_total += loss
            labels = torch.max(labels.data, 1)[1].cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        # print("input_ids_all:", input_ids_all.shape)
        # print("attention_mask_all:", attention_mask_all.shape)
        # print("segment_ids:", segment_ids_all.shape)
        # print("outputs_all shape:", outputs_all.shape)
        # print("labels_all shape:", labels_all.shape)
        # print("predict_all shape:", predict_all.shape)
        with open(config_inf.output_path, 'w', encoding='utf-8') as file:
            index = 0
            for i in range(len(input_ids_all)):
                input_ids_list = convert_numpy_int_to_python_int(input_ids_all[i].tolist())
                attention_mask_list = convert_numpy_int_to_python_int(attention_mask_all[i].tolist())
                segment_ids_list = convert_numpy_int_to_python_int(segment_ids_all[i].tolist())
                data = {
                    'index':index,
                    'text': texts_all[i],
                    'input_ids':input_ids_list,
                    'attention_mask':attention_mask_list,
                    'segment_ids':segment_ids_list,
                    'target':int(labels_all[i]), 'predict':int(predict_all[i])
                }
                outputs = outputs_all[i]
                for j in range(config_inf.num_classes):
                    data[config_inf.class_list[j]] = outputs[j]
                json.dump(data, file, ensure_ascii=False)
                file.write('\n')
                index += 1

        # 生成分类报告
        report = metrics.classification_report(labels_all, predict_all, target_names=config_inf.class_list, digits=4)
        # 生成混淆矩阵
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_loader), report, confusion
    return acc, loss_total / len(data_loader)

