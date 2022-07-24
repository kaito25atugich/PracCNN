import time

import torch

from test import test
from train import train



def eval_train(body, device, train_data_loader, test_data_loader):
    print("train mode")
    body.history["train_loss"] = []
    body.history["train_accuracy"] = []

    start = time.perf_counter()
    for epoch in range(body.n_epochs):
        train_loss, train_accuracy = train(body.model, device, train_data_loader, body.opt, body.loss_func)
        body.history["train_loss"].append(train_loss)
        body.history["train_accuracy"].append(train_accuracy)
        test_loss, test_accuracy = test(body.model, device, test_data_loader, body.loss_func)
        body.history["test_loss"].append(test_loss)
        body.history["test_accuracy"].append(test_accuracy)

        print(f"#### epoch {epoch + 1} ##### \ntrain loss: {train_loss:.6f}, train accuracy: {train_accuracy:.0%}")
        print(f"test loss: {test_loss:.6f}, test accuracy: {test_accuracy:.0%}")

    end = time.perf_counter()
    print(f"train end. time {end - start} (s)")

    model_path = "./model/model.pth"
    torch.save(body.model.state_dict(), model_path)