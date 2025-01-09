import numpy as np
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
# import torchmetrics
import json



def convert_numpy_int_to_python_int(lst):
    return [int(x) if isinstance(x, np.int64) else x for x in lst]



def evaluate(model, data_loader, test=False):
    config_inf = config.Config()
    model.eval()

    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    texts_all = np.array([], dtype=str)
    input_ids_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    attention_mask_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    segment_ids_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    outputs_all = np.array([[]], dtype=np.float64).reshape(0, config_inf.num_classes)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels, texts in data_loader:
            texts_all = np.append(texts_all, texts)
            input_ids_all = np.append(input_ids_all, inputs[0].cpu().numpy(), axis = 0)
            segment_ids_all = np.append(segment_ids_all, inputs[1].cpu().numpy(), axis = 0)
            attention_mask_all = np.append(attention_mask_all, inputs[2].cpu().numpy(), axis = 0)

            logits = model(inputs)
            outputs = softmax(logits)

            outputs_all = np.append(outputs_all, outputs.cpu().numpy(), axis = 0)
            
            loss = F.cross_entropy(logits, labels)  # 交叉熵损失
            loss_total += loss

            labels = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
        accuracy = metrics.accuracy_score(labels_all, predict_all)

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
            
        report = metrics.classification_report(labels_all, predict_all, target_names=config_inf.class_list, digits=4)
        # 生成混淆矩阵
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return accuracy, loss_total / len(data_loader), report, confusion
    return accuracy, loss_total / len(data_loader)

