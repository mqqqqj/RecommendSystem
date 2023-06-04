# MatrixFactorization FunkSVD:
# machine learning based matrix factorization optimizing prediction accuracy with MSE.
import numpy as np
import time
import pickle
from matplotlib import pyplot as plt
import os


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
        self.best_rmse = 100

    def train(self, train_data, valid_data, EPOCH, FOLD):
        init_rmse = self.RMSE(valid_data)
        print(f"{date()}## Before training, valid rmse is:{init_rmse:.6f}")
        print(f"{date()}## Start training!")
        start_time = time.perf_counter()
        rmse_list = [init_rmse]
        for epoch in range(EPOCH):
            for userID, items in train_data.items():
                for itemID in items.keys():
                    r_ui = items[itemID]
                    r_ui_h = (
                        self.global_mean
                        + self.user_bias[userID]
                        + self.item_bias[itemID]
                        + np.dot(self.pu[userID], self.qi[itemID])
                    )
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
            train_rmse = self.RMSE(train_data)
            valid_rmse = self.RMSE(valid_data)
            rmse_list.append(valid_rmse)
            end_time = time.perf_counter()
            print(
                f"{date()}#### Epoch {epoch:3d}: rmse on train set is {train_rmse:.6f}, rmse on valid set is {valid_rmse:.6f},costs {end_time - start_time:.0f} seconds totally."
            )
            if valid_rmse < self.best_rmse:
                self.best_rmse = valid_rmse
                self.save()
        self.draw_rmse(FOLD, rmse_list)

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

    def draw_rmse(self, fold, rmse_list):
        plt.switch_backend("Agg")
        plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
        plt.plot(rmse_list, "b", label="rmse")
        plt.ylabel("ValidSet RMSE")
        plt.xlabel("EPOCH")
        plt.legend()  # 个性化图例（颜色、形状等）
        save_path = "./results/fold_" + str(fold) + ".png"
        plt.savefig(save_path)
