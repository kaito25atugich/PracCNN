import time

import torch

def train(model, device, data_loader, opt, loss_f):
    """
    1エポックの訓練
    """

    # モデルを訓練モードに
    model.train()

    running_loss = 0
    total_correct = 0

    for data, target in  data_loader:        
        data ,target = data.to(device), target.to(device)

        # 順伝搬
        output = model(data)

        # 損失関数の値を計算
        loss = loss_f(output, target) 
        running_loss += float(loss)

        # 逆伝搬        
        opt.zero_grad()
        loss.backward()

        # パラメータの更新
        opt.step()

        # 予測
        predict = output.argmax(dim=1) 
        total_correct += int((predict == target).sum())

    avg_loss = running_loss / len(data_loader.dataset)
    acc = total_correct / len(data_loader.dataset)

    return avg_loss, acc

