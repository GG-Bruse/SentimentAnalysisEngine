import numpy as np
import config
import torch
import torch.nn as nn
import torchmetrics
import json



def calculation_accuracy(config_inf, true, predic):
    true = torch.argmax(true, dim=1)
    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=config_inf.num_classes)
    accuracy.update(predic, true)
    return accuracy.compute()

def calculation_f1(config_inf, true, predic):
    true = torch.argmax(true, dim=1)
    f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=config_inf.num_classes)
    f1_score.update(predic, true)
    return f1_score.compute()



def convert_numpy_int_to_python_int(lst):
    return [int(x) if isinstance(x, np.int64) else x for x in lst]



def evaluate(model, data_loader, test=False):
    config_inf = config.Config()
    model.eval()
    criterion = nn.BCELoss()

    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    texts_all = np.array([], dtype=str)
    input_ids_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    attention_mask_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    segment_ids_all = np.array([[]], dtype=int).reshape(0, config_inf.max_sequence_length)
    outputs_all = np.array([[]], dtype=np.float64).reshape(0, config_inf.num_classes)

    outputs_all_tensor = torch.empty(0, dtype=torch.float32).to(config_inf.device)
    labels_all_tensor = torch.empty(0, dtype=torch.float32).to(config_inf.device)

    with torch.no_grad():
        for inputs, labels, texts in data_loader:
            texts_all = np.append(texts_all, texts)
            input_ids_all = np.append(input_ids_all, inputs[0].cpu().numpy(), axis = 0)
            segment_ids_all = np.append(segment_ids_all, inputs[1].cpu().numpy(), axis = 0)
            attention_mask_all = np.append(attention_mask_all, inputs[2].cpu().numpy(), axis = 0)
            outputs = model(inputs)
            outputs_all = np.append(outputs_all, outputs.cpu().numpy(), axis = 0)
            
            outputs_all_tensor = torch.cat((outputs_all_tensor, outputs), dim=0)
            labels_all_tensor = torch.cat((labels_all_tensor, labels), dim=0)

            loss = criterion(outputs, labels)
            loss_total += loss

            labels = torch.max(labels.data, 1)[1].cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

        true = labels_all_tensor.data.cpu()
        predic = outputs_all_tensor.data.cpu()
        accuracy = calculation_accuracy(config_inf, true, predic)

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
        f1 = calculation_f1(config_inf, true, predic)
        
        return (loss_total / len(data_loader)).cpu(), accuracy, f1
    return loss_total / len(data_loader), accuracy

