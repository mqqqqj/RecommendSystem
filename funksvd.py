# MatrixFactorization FunkSVD:
# machine learning based matrix factorization optimizing prediction accuracy with MSE.
import torch
import torch.nn as nn
import numpy as np
import time
import pickle

# class FunkSVD(nn.Module):
#     def __init__(self, M, N, K=100):
#         super().__init__()
#         self.user_emb = nn.Parameter(torch.randn(M, K))
#         self.user_bias = nn.Parameter(torch.randn(M))  # 偏置
#         self.item_emb = nn.Parameter(torch.randn(N, K))
#         self.item_bias = nn.Parameter(torch.randn(N))
#         self.bias = nn.Parameter(torch.zeros(1))  # 全局偏置

#     def forward(self, user_id, item_id):
#         pred = self.user_emb[user_id] * self.item_emb[item_id]
#         pred = (
#             pred.sum(dim=-1)
#             + self.user_bias[user_id]
#             + self.item_bias[item_id]
#             + self.bias
#         )
#         return pred


def date(f="%Y-%m-%d %H:%M:%S"):
    return time.strftime(f, time.localtime())


class FunkSVD:
    def __init__(self, M=19835, N=624961, K=100):
        super().__init__()
        self.user_bias = np.zeros(M)  # 用户偏置
        self.item_bias = np.zeros(N)  # 商品偏置
        self.pu = np.random.rand(M, K)
        self.qi = np.random.rand(N, K)
        self.global_mean = 49.50457011488369  # 从data_anaylisis.py 得到
        self.lr = 0.0005  # 学习率
        self.l = 0.02  # 正则化系数

    def train(self, data, EPOCH):
        best_rmse = self.RMSE(data)
        print(f"{date()}## Before training, rmse is:{best_rmse:.6f}")
        print(f"{date()}## Start training!")
        start_time = time.perf_counter()
        for epoch in range(EPOCH):
            for userID, items in data.items():
                for itemID in items.keys():
                    r_ui = items[itemID]
                    r_ui_h = (
                        self.global_mean
                        + self.user_bias[userID]
                        + self.item_bias[itemID]
                        + np.dot(self.pu[userID], self.qi[itemID])
                    )
                    # self.backward(
                    #     label=r_ui, predict=r_ui_h, userID=userID, itemID=itemID
                    # )
                    loss = r_ui - r_ui_h
                    # print(loss)
                    if np.isnan(loss):
                        exit()
                    self.user_bias[userID] += self.lr * (
                        loss - self.l * self.user_bias[userID]
                    )
                    self.item_bias[itemID] += self.lr * (
                        loss - self.l * self.item_bias[itemID]
                    )
                    old_pu = self.pu[userID]
                    self.pu[userID] += self.lr * (
                        loss * self.qi[itemID] - self.l * old_pu
                    )
                    self.qi[itemID] += self.lr * (
                        loss * old_pu - self.l * self.qi[itemID]
                    )
            rmse = self.RMSE(data)
            end_time = time.perf_counter()
            print(
                f"{date()}#### Epoch {epoch:3d}: rmse {rmse:.6f}, costs {end_time - start_time:.0f} seconds totally."
            )
            if rmse < best_rmse:
                best_rmse = rmse
                self.save()

    def backward(self, label, predict, userID, itemID):
        loss = label - predict
        self.user_bias[userID] += self.lr * (loss - self.l * self.user_bias[userID])
        self.item_bias[itemID] += self.lr * (loss - self.l * self.item_bias[itemID])
        old_pu = self.pu[userID]
        print(loss)
        if np.isnan(loss):
            exit()
        self.pu[userID] += self.lr * (loss * self.qi[itemID] - self.l * old_pu)
        self.qi[itemID] += self.lr * (loss * old_pu - self.l * self.qi[itemID])

    def RMSE(self, data):
        sum = 0
        num = 0
        for userID, items in data.items():
            for itemID in items.keys():
                r_ui = items[itemID]
                r_ui_h = (
                    self.global_mean
                    + self.user_bias[int(userID)]
                    + self.item_bias[int(itemID)]
                    + np.dot(self.pu[int(userID)], self.qi[int(itemID)])
                )
                sum += (r_ui - r_ui_h) ** 2
                num += 1
        return np.sqrt(sum / num)

    def save(self):
        with open("./models/funkSVD.pkl", "wb") as f:
            pickle.dump(self, f)
