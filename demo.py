import os
import math
import pickle
import copy
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from my_dataset import MyDataset
from funksvd import FunkSVD

train_path = "./data/train.txt"

test_path = "./data/test.txt"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using PyTorch version:", torch.__version__, " Device:", device)


BATCH_SIZE = 256
EPOCHS = 10


def process_data(src_path, dumped_path):
    """
    no dumped path, process data.
    """
    data = dict()
    with open(src_path, "rb") as r_file:
        line = r_file.readline()
        while line:
            user_id, n_item = line.decode().split("|")
            user_id = int(user_id)
            item_score_list = dict()
            for _ in range(int(n_item)):
                line = r_file.readline()
                item_id, score = line.decode().split("  ")
                item_score_list[int(item_id)] = int(score)
            data[int(user_id)] = item_score_list
            line = r_file.readline()
    with open(dumped_path, "wb") as w_file:
        pickle.dump(data, w_file)
    return data


def get_train_data(dumped_path):
    """
    get train data
    """
    if os.path.exists(dumped_path):
        print("train data has processed.")
        with open(dumped_path, "rb") as f:
            data = pickle.load(f)
            print("data load from pickle succeed.")
            return data
    else:
        print("Process data from txt.")
        data = process_data(train_path, dumped_path)
        return data


def data2csv(data):
    df = pd.DataFrame(data=data, columns=["userID", "itemID", "score"])
    df.to_csv("train.csv", index=False, header=False)


def preprocess(train_path):
    # [user_id, item_id, score]
    data = []
    with open(train_path, "rb") as r_file:
        line = r_file.readline()
        while line:
            user_id, n_item = line.decode().split("|")
            user_id = int(user_id)
            for _ in range(int(n_item)):
                line = r_file.readline()
                item_id, score = line.decode().split("  ")
                temp = [user_id, int(item_id), int(score)]
                data.append(temp)
            line = r_file.readline()
    df = pd.DataFrame(data=data, columns=["userID", "itemID", "score"])
    df.to_csv("train.csv", index=False, header=False)


def RMSE(model, dataloader):
    rmse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_id, item_id, scores = [
                i.to(next(model.parameters()).device) for i in batch
            ]
            predict = model(user_id, item_id)
            rmse += torch.nn.functional.mse_loss(
                predict, scores, reduction="sum"
            ).item()  # mean
            sample_count += len(scores)
    rmse /= sample_count
    rmse = math.sqrt(rmse)
    return rmse


def train(model, train_set, valid_set):
    print("calculate init rmse.")
    train_rmse = RMSE(model, train_set)
    valid_rmse = RMSE(model, valid_set)
    print(
        "before starting training: rmse on train set is:",
        train_rmse,
        ", rmse on valid set is:",
        valid_rmse,
    )
    print("start training.")
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    # lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.99)  # 学习率衰减
    best_loss = 100
    for epoch in range(10):
        model.train()
        total_loss, total_samples = 0, 0
        for batch in train_set:
            user_id, item_id, ratings = [i.to(device) for i in batch]
            # print(user_id, item_id, ratings)
            # exit()
            predict = model(user_id, item_id)
            print(predict, ratings)
            # exit()
            RMSELoss = torch.sqrt(F.mse_loss(predict, ratings, reduction="mean"))
            opt.zero_grad()
            RMSELoss.backward()
            opt.step()

            total_loss += RMSELoss.item() * len(predict)
            total_samples += len(predict)

        # lr_sch.step()
        model.eval()  # 停止训练状态
        valid_rmse = RMSE(model, valid_set)
        train_loss = total_loss / total_samples
        print(
            f"#### Epoch {epoch:3d}; train rmse {train_loss:.6f}; validation rmse {valid_rmse:.6f}"
        )

        if best_loss > valid_rmse:
            best_loss = valid_rmse
            torch.save(model, "./models/funksvd.pt")


# 16199,414384,70
if __name__ == "__main__":
    if os.path.exists("./data/train.csv") == False:
        print("csv not exist.")
        preprocess(train_path)
    # 划分训练集：验证集：测试集 = 8：2
    df = pd.read_csv("./data/train.csv", usecols=[0, 1, 2])
    df.columns = ["userID", "itemID", "score"]
    train_data, valid_data = train_test_split(df, test_size=0.2, random_state=3)
    train_set = DataLoader(
        dataset=MyDataset(train_data), batch_size=BATCH_SIZE, shuffle=True
    )
    valid_set = DataLoader(
        dataset=MyDataset(valid_data), batch_size=BATCH_SIZE, shuffle=False
    )
    model = FunkSVD(19835, 624960).to(device=device)
    print(model)
    train(model, train_set, valid_set)
