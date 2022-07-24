from audioop import avg
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import torch
import torchvision

from fashion_model import FashionModel

def test(model, device, data_loader, loss_f):
    model.eval()

    with torch.no_grad():
        running_loss = 0
        total_correct = 0 
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = loss_f(output, target)
            running_loss += float(loss)

            predict = output.argmax(dim=1) 
            total_correct += int((predict == target).sum())

    avg_loss = running_loss / len(data_loader.dataset)
    acc = total_correct / len(data_loader.dataset)

    return avg_loss, acc

def test_f_eval(model, device, data_loader, loss_f):
    model.eval()
    
    with torch.no_grad():
        running_loss = 0
        total_correct = 0 

        targets = []
        predicts = []
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            targets += target.tolist()
            loss = loss_f(output, target)
            running_loss += float(loss)

            predict = output.argmax(dim=1) 
            predicts += predict.tolist()
            total_correct += int((predict == target).sum())

    avg_loss = running_loss / len(data_loader.dataset)
    acc = total_correct / len(data_loader.dataset)
    print(f"avg_loss:{avg_loss}, acc:{acc}") 
    pred = np.array(predicts)
    targ = np.array(targets)
    CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    cm = confusion_matrix(targ, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("./eval/conf.png")
    AVG = "macro"
    accuracy = accuracy_score(targ, pred)
    precision = precision_score(targ, pred, average=AVG)
    recall = recall_score(targ, pred, average="macro")
    f1 = f1_score(targ, pred, average="macro")
    print(f"accuracy: {accuracy}  precision: {precision}  recall: {recall}  f1: {f1}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/model.pth"
    data_path = "./data/"
    model = FashionModel().to(device)
    model.load_state_dict(torch.load(model_path))

    test_data = torchvision.datasets.FashionMNIST(data_path, train=False, transform=torchvision.transforms.ToTensor(), download=False)
    
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    loss_func = torch.nn.NLLLoss()

    test_f_eval(model, device, test_data_loader, loss_func)
