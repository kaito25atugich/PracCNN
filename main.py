from collections import defaultdict
from re import I
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from fashion_model import FashionModel
from make_graph import draw_all
from test import test
from evaluation import eval_train


class TrainSystem:
    def __init__(self, device, n_epochs=50):
        # モデルを転送
        self.model = FashionModel().to(device)
        self.n_epochs = n_epochs
        # 損失関数
        self.loss_func = torch.nn.NLLLoss()
        # 最適化手法
        self.opt = torch.optim.Adam(self.model.parameters())
        self.history = defaultdict(list)

def main():
    data_path = "./data/"

    # 訓練データの読み込み 
    # train: 訓練データorテストデータどちらをロード、transform:データの前処理\ totensor float32型で[0,1]に正規化、画素の範囲が[0,255]
    train_data = torchvision.datasets.FashionMNIST(data_path, train=True, transform=torchvision.transforms.ToTensor(), download=False)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # テストデータの読み込み
    test_data = torchvision.datasets.FashionMNIST(data_path, train=False, transform=torchvision.transforms.ToTensor(), download=False)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    # 計算リソースの選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 訓練の実施
    n_epochs = 50 

    body = TrainSystem(device, n_epochs)
    
    while True:
        print("menuを入力(h/Hでヘルプ):")
        menu = input()
        if menu == "h" or menu == "H":
            print("h/H: ヘルプ\nq/Q: 終了\n0: 訓練モード\n4: モデルを読み込みテストモード\n9: システム変更")
        elif menu == "0":
            eval_train(body, device, train_data_loader, test_data_loader)
            draw_all(body)
            test(body.model, device, test_data_loader, body.loss_func)
        elif menu == "4":
            print("モデルのパスを入力")
            model_path = input()
            try:
                body.model.load_state_dict(torch.load(model_path))
            except:
                print("パスが存在しません")
            test(body.model, device, test_data_loader, body.loss_func)
        elif menu == "q" or menu == "Q":
            break

if __name__ == "__main__":
    main()