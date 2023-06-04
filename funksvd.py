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
    def __init__(self, FOLD, M=19835, N=624961, K=100):
        super().__init__()
        self.user_bias = np.zeros(M)  # 用户偏置
        self.item_bias = np.zeros(N)  # 商品偏置
        self.pu = np.random.rand(M, K)
        self.qi = np.random.rand(N, K)
        self.global_mean = 0  # 从data_anaylisis.py 得到
        self.lr = 0.0005  # 学习率
        self.l = 0.02  # 正则化系数
        self.best_rmse = 100
        self.save_path = "./models/funkSVD_" + str(FOLD) + ".pkl"

    def train(self, train_data, valid_data, EPOCH, FOLD):
        self.global_mean = self.set_global_mean(train_data)
        print(
            f"{date()}## Before training, init global mean score:{self.global_mean:.6f}"
        )
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
                    self.backward(
                        label=r_ui, predict=r_ui_h, userID=userID, itemID=itemID
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

    def set_global_mean(self, train_data):
        avg = 0
        num = 0
        for _, items in train_data.items():
            for itemID in items.keys():
                avg += items[itemID]
                num += 1
        avg /= num
        return avg

    def backward(self, label, predict, userID, itemID):
        loss = label - predict
        self.user_bias[userID] += self.lr * (loss - self.l * self.user_bias[userID])
        self.item_bias[itemID] += self.lr * (loss - self.l * self.item_bias[itemID])
        old_pu = self.pu[userID]
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
                    + self.user_bias[userID]
                    + self.item_bias[itemID]
                    + np.dot(self.pu[userID], self.qi[itemID])
                )
                sum += (r_ui - r_ui_h) ** 2
                num += 1
        return np.sqrt(sum / num)

    def save(self):
        with open(self.save_path, "wb") as f:
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

    def predict(self, test_data):
        with open("./results/result.txt", "w") as w_file:
            for userID, itemlist in test_data.items():
                w_file.write(str(userID) + "|" + str(itemlist[0]) + "\n")
                for i in range(itemlist[0]):
                    itemID = itemlist[i + 1]
                    r_ui_h = (
                        self.global_mean
                        + self.user_bias[userID]
                        + self.item_bias[itemID]
                        + np.dot(self.pu[userID], self.qi[itemID])
                    )
                    w_file.write(str(itemID) + "  " + str(r_ui_h) + "\n")


# print(test_data[0])
# with open("./results/result.txt", "w") as w_file, open(
#     "./data/test.txt", "r"
# ) as r_file:
#     line = r_file.readline()
#     while line:
#         w_file.write(line)
#         userID, n_item = line.split("|")
#         for _ in range(int(n_item)):
#             itemID = r_file.readline()
#             r_ui_h = (
#                 self.global_mean
#                 + self.user_bias[int(userID)]
#                 + self.item_bias[int(itemID)]
#                 + np.dot(self.pu[int(userID)], self.qi[int(itemID)])
#             )
#             w_file.write(userID + "  " + itemID + "\n")
#         line = r_file.readline()
