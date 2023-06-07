# MatrixFactorization FunkSVD:
# machine learning based matrix factorization optimizing prediction accuracy with MSE.
import os

import numpy as np
import time
import math
import pickle
from matplotlib import pyplot as plt


def date(f="%Y-%m-%d %H:%M:%S"):
    return time.strftime(f, time.localtime())


class FunkSVD:
    def __init__(self, FOLD, M=19835, N=624961, K=100, optim=False):
        super().__init__()
        self.user_bias = np.zeros(M)  # 用户偏置
        self.item_bias = np.zeros(N)  # 商品偏置
        self.pu = np.random.rand(M, K)
        self.qi = np.random.rand(N, K)
        self.global_mean = 0  # 从data_anaylisis.py 得到
        self.lr = 0.0005  # 学习率
        self.l = 0.02  # 正则化系数
        self.best_rmse = 100
        self.opt_method = "cos"
        self.optim = optim
        self.fold = FOLD
        self.cos_dump_path = "./data/cos_simi_" + str(FOLD) + ".pkl"
        self.euc_dump_path = "./data/euc_simi_" + str(FOLD) + ".pkl"
        if self.optim:
            self.save_path = "./models/Opt_"+ self.opt_method+ "FunkSVD_" + str(FOLD) + ".pkl"
            self.N_neighbors = 5
        else:
            self.save_path = "./models/funkSVD_" + str(FOLD) + ".pkl"
            self.N_neighbors = 0

    def train(self, train_data, valid_data, EPOCH, FOLD):
        """
        训练模型，train_data 用作训练，valid_data用作验证
        """
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
            if self.optim:
                train_rmse = self.opt_RMSE(train_data, False)
                valid_rmse = self.opt_RMSE(valid_data, True)
            else:
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
        self.draw_rmse(FOLD, rmse_list, EPOCH)

    def set_global_mean(self, train_data):
        """
        获取train_data上的评分平均值
        """
        avg = 0
        num = 0
        for _, items in train_data.items():
            for itemID in items.keys():
                avg += items[itemID]
                num += 1
        avg /= num
        return avg

    def backward(self, label, predict, userID, itemID):
        """
        更新矩阵参数
        """
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

    def opt_RMSE(self, data, is_valid=False):
        similarity = {}
        if is_valid and self.opt_method == "cos" and os.path.exists(self.cos_dump_path):
            with open(self.cos_dump_path, "rb") as f:
                similarity = pickle.load(f)
        if is_valid and self.opt_method == "euc" and os.path.exists(self.euc_dump_path):
            with open(self.euc_dump_path, "rb") as f:
                similarity = pickle.load(f)
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
                if is_valid and userID in similarity.keys():
                    item_simi = similarity[userID]
                    if itemID in item_simi.keys():
                        smi_rate = 0.2
                        similarity_score = item_simi[itemID]
                        if similarity_score == 0:
                            smi_rate = 0
                        r_ui_h = r_ui_h * (1 - smi_rate) + similarity_score * smi_rate
                sum += (r_ui - r_ui_h) ** 2
                num += 1
        return np.sqrt(sum / num)

    def save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)

    def draw_rmse(self, fold, rmse_list, epoch):
        plt.switch_backend("Agg")
        plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
        plt.plot(rmse_list, "b", label="rmse")
        plt.ylabel("ValidSet RMSE")
        plt.xlabel("EPOCH")
        plt.legend()  # 个性化图例（颜色、形状等）
        if self.optim:
            save_path = (
                "./results/"
                + self.opt_method
                + "FunkSVD_epo"
                + str(epoch)
                + "_fold_"
                + str(fold)
                + ".png"
            )
        else:
            save_path = (
                "./results/FunkSVD_epo" + str(epoch) + "_fold_" + str(fold) + ".png"
            )
        plt.savefig(save_path)

    def predict(self, train_data, test_data):
        if self.optim == False:
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
                        r_ui_h = min(100, max(0, r_ui_h))
                        w_file.write(str(itemID) + "  " + str(r_ui_h) + "\n")
        else:
            with open("./data/attr.pkl", "rb") as r_file:
                item_attribute = pickle.load(r_file)
            with open(
                "./results/result_" + str(self.fold) + "_" + self.opt_method + ".txt",
                "w",
            ) as w_file, open(
                "./results/" + str(self.fold) + "_" + self.opt_method + "_res.txt", "w"
            ) as w2f:
                for userID, itemlist in test_data.items():
                    w_file.write(str(userID) + "|" + str(itemlist[0]) + "\n")
                    w2f.write(str(userID) + "|" + str(itemlist[0]) + "\n")
                    for i in range(int(itemlist[0])):
                        itemID = itemlist[i + 1]
                        r_ui_h = (
                                self.global_mean
                                + self.user_bias[userID]
                                + self.item_bias[itemID]
                                + np.dot(self.pu[userID], self.qi[itemID])
                        )
                        pred = r_ui_h
                        similarity_score = 0
                        if itemID in item_attribute.keys():
                            smi_rate = 0.4
                            if self.opt_method == "cos":
                                similarity_score = self.get_cos_simi_score(
                                    item_attribute, train_data, userID, itemID
                                )
                            elif self.opt_method == "euc":
                                similarity_score = self.get_euc_simi_score(
                                    item_attribute, train_data, userID, itemID
                                )
                            if similarity_score == 0:
                                smi_rate = 0
                            r_ui_h = r_ui_h * (1 - smi_rate) + similarity_score * smi_rate
                        r_ui_h = min(100, max(0, r_ui_h))
                        w_file.write(str(itemID) + "  " + str(r_ui_h) + "\n")
                        w2f.write(
                            str(itemID) + "  " + str(r_ui_h) + "  opt:" + str(similarity_score) + "  pred:" + str(
                                pred) + "\n")

    def cosine_similarity(self, item_attribute, item_a, item_b):
        attr_a = item_attribute[item_a]
        attr_b = item_attribute[item_b]
        mo_a = math.sqrt(attr_a[0] ** 2 + attr_a[1] ** 2)
        mo_b = math.sqrt(attr_b[0] ** 2 + attr_b[1] ** 2)
        res = np.dot(attr_a, attr_b) / (mo_a * mo_b)
        return res

    def get_cos_simi_score(self, item_attribute, train_data, userID, itemID):
        items = train_data[userID]
        similarity_dict = dict()
        for item in items.keys():
            if item not in item_attribute.keys():
                continue
            if item_attribute[item][0] == -1 or item_attribute[item][1] == -1:
                continue
            cos = self.cosine_similarity(item_attribute, itemID, item)
            if cos > 0.5:
                similarity_dict[item] = cos
        similarity_list = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=False)
        score = 0
        simi = 0
        for i in range(min(self.N_neighbors, len(similarity_list))):
            score += train_data[userID][similarity_list[i][0]] * similarity_list[i][1]
            simi += similarity_list[i][1]
        if simi == 0:
            return 0
        score = score / simi
        return score

    def dump_valid_cos_simi(self, valid_data, train_data):
        with open("./data/attr.pkl", "rb") as r_file:
            item_attribute = pickle.load(r_file)
        cos_simi = dict()
        count = 0
        for userID, items in valid_data.items():
            cos_simi[userID] = dict()
            for itemID in items.keys():
                if itemID in item_attribute.keys():
                    cos_similarity_score = self.get_cos_simi_score(
                        item_attribute, train_data, userID, itemID
                    )
                    cos_simi[userID][itemID] = cos_similarity_score
            count += 1
            if count % 2000 == 0:
                count = 0
                print("continue, ", userID)
        with open(self.cos_dump_path, "wb") as w_file:
            pickle.dump(cos_simi, w_file)
        return cos_simi

    def euclidean_distance_similarity(self, item_attribute, item_a, item_b):
        attr_a = item_attribute[item_a]
        attr_b = item_attribute[item_b]
        sq_1 = (attr_a[0] - attr_b[0]) ** 2
        sq_2 = (attr_a[1] - attr_b[1]) ** 2
        res = math.sqrt(sq_1 + sq_2)
        return res

    def get_euc_simi_score(self, item_attribute, train_data, userID, itemID):
        items = train_data[userID]
        similarity_dict = dict()
        for item in items.keys():
            if item not in item_attribute.keys():
                continue
            if item_attribute[item][0] == -1 or item_attribute[item][1] == -1:
                continue
            distance = self.euclidean_distance_similarity(item_attribute, itemID, item)
            if distance < 1000:
                similarity_dict[item] = distance
        similarity_list = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
        score = 0
        simi = 0
        for i in range(min(self.N_neighbors, len(similarity_list))):
            score += train_data[userID][similarity_list[i][0]]
            simi += 1
        if simi == 0:
            return 0
        score = score / simi
        return score

    def dump_valid_euc_simi(self, valid_data, train_data):
        with open("./data/attr.pkl", "rb") as r_file:
            item_attribute = pickle.load(r_file)
        euc_simi = dict()
        count = 0
        for userID, items in valid_data.items():
            euc_simi[userID] = dict()
            for itemID in items.keys():
                if itemID in item_attribute.keys():
                    euc_similarity_score = self.get_euc_simi_score(
                        item_attribute, train_data, userID, itemID
                    )
                    euc_simi[userID][itemID] = euc_similarity_score
            count += 1
            if count % 2000 == 0:
                count = 0
                print("continue, ", userID)
        with open(self.euc_dump_path, "wb") as w_file:
            pickle.dump(euc_simi, w_file)
        return euc_simi
