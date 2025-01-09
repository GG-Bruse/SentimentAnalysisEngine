# coding: UTF-8
import torch
import time
from evaluate import evaluate

def test(model, test_data_loader, path):
    model.load_state_dict(torch.load(path, weights_only=True))
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