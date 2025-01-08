# coding: UTF-8
import torch
import time
from evaluate import evaluate

def test(model, test_data_loader, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    start_time = time.time()

    test_loss, accuracy, f1 = evaluate(model, test_data_loader, test=True)
    print("Loss...")
    print(test_loss)
    print("accuracy...")
    print(accuracy)
    print("f1...")
    print(f1)

    end_time = time.time()
    print("Time usage:", end_time - start_time)