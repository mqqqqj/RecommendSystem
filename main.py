import os
import copy
import math
import pickle
import numpy as np
from funksvd import FunkSVD

train_path = "./data/train.txt"

test_path = "./data/test.txt"

attr_path = "./data/itemAttribute.txt"

train_dump_path = "./data/train.pkl"

test_dump_path = "./data/test.pkl"

attr_dump_path = "./data/attr.pkl"

EPOCH = 20
K = 100
N_folds = 5


def dump_train_data(src_path, dumped_path):
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


def dump_test_data(src_path, dumped_path):
    data = dict()
    with open(src_path, "rb") as r_file:
        line = r_file.readline()
        while line:
            user_id, n_item = line.decode().split("|")
            user_id = int(user_id)
            data[int(user_id)] = [int(n_item)]
            for _ in range(int(n_item)):
                item_id = r_file.readline().decode()
                data[int(user_id)].append(int(item_id))
            line = r_file.readline()
    with open(dumped_path, "wb") as w_file:
        pickle.dump(data, w_file)
    return data


def dump_attr(src_path, dumped_path):
    item_attribute = dict()
    with open(src_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            itemID, attr_1, attr_2 = line.strip().split("|")
            attr_1 = int(attr_1) if attr_1 != "None" else -1
            attr_2 = int(attr_2) if attr_2 != "None" else -1
            # item_attribute[int(itemID)] = np.array([attr_1, attr_2])
            item_attribute[int(itemID)] = [attr_1, attr_2]
    with open(dumped_path, "wb") as w_file:
        pickle.dump(item_attribute, w_file)
    return item_attribute


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
        data = dump_train_data(train_path, dumped_path)
        return data


def get_test_data(dumped_path):
    """
    get test data
    """
    if os.path.exists(dumped_path):
        print("test data has processed.")
        with open(dumped_path, "rb") as f:
            data = pickle.load(f)
            print("data load from pickle succeed.")
            return data
    else:
        print("Process data from txt.")
        data = dump_test_data(test_path, dumped_path)
        return data


def get_attr(dumped_path):
    """
    get attr data
    """
    if os.path.exists(dumped_path):
        print("attr has processed.")
        with open(dumped_path, "rb") as f:
            data = pickle.load(f)
            print("attr load from pickle succeed.")
            return data
    else:
        print("Process attr from txt.")
        data = dump_attr(attr_path, dumped_path)
        return data


def cross_val_score(all_data, n_folds, fold):
    """
    根据传入的折数划分训练集和验证集
    """
    train_data = copy.deepcopy(all_data)
    valid_data = {}
    for userID, items in all_data.items():
        n_items = len(items)
        keys = list(items.keys())
        chunk_size = math.ceil(n_items / n_folds)
        tempdict = dict()
        for j in range(fold * chunk_size, min((fold + 1) * chunk_size, n_items)):
            tempdict[keys[j]] = items[keys[j]]
            train_data[userID].pop(keys[j])
        valid_data[userID] = tempdict
    return train_data,valid_data


def avg_predict(data, model_list, is_valid):
    """
    在alldata上对五个模型的预测结果取均值,进行rmse/opt_rmse的计算
    """
    if model_list[0].optim:
        print("funksvd with" + model_list[0].opt_method)
        similarity = [{}, {}, {}, {}, {}]
        for i in range(N_folds):
            if (
                is_valid
                and model_list[i].opt_method == "cos"
                and os.path.exists(model_list[i].cos_dump_path)
            ):
                with open(model_list[i].cos_dump_path, "rb") as f:
                    similarity[i] = pickle.load(f)
            if (
                is_valid
                and model_list[i].opt_method == "euc"
                and os.path.exists(model_list[i].euc_dump_path)
            ):
                with open(model_list[i].euc_dump_path, "rb") as f:
                    similarity[i] = pickle.load(f)
        sum = 0
        num = 0
        for userID, items in data.items():
            for itemID in items.keys():
                r_ui = items[itemID]
                r_ui_h = [0, 0, 0, 0, 0]
                for i in range(N_folds):
                    r_ui_h[i] = (
                        model_list[i].global_mean
                        + model_list[i].user_bias[userID]
                        + model_list[i].item_bias[itemID]
                        + np.dot(model_list[i].pu[userID], model_list[i].qi[itemID])
                    )
                    if is_valid and userID in similarity[i].keys():
                        item_simi = similarity[i][userID]
                        if itemID in item_simi.keys():
                            smi_rate = 0.3
                            similarity_score = item_simi[itemID]
                            if similarity_score == 0:
                                smi_rate = 0
                            r_ui_h[i] = (
                                r_ui_h[i] * (1 - smi_rate) + similarity_score * smi_rate
                            )
                avg_score = np.mean(r_ui_h)
                sum += (r_ui - avg_score) ** 2
                num += 1
        return np.sqrt(sum / num)
    else:
        print("basic funksvd")
        sum = 0
        num = 0
        for userID, items in data.items():
            for itemID in items.keys():
                r_ui = items[itemID]
                r_ui_h = [0, 0, 0, 0, 0]
                for i in range(N_folds):
                    r_ui_h[i] = (
                        model_list[i].global_mean
                        + model_list[i].user_bias[userID]
                        + model_list[i].item_bias[itemID]
                        + np.dot(model_list[i].pu[userID], model_list[i].qi[itemID])
                    )
                avg_score = np.mean(r_ui_h)
                sum += (r_ui - avg_score) ** 2
                num += 1
        return np.sqrt(sum / num)


def optfunksvd_avg_test(test_data, model_list):
    """
    使用优化算法对测试集进行评分
    """
    assert model_list[0].optim is True
    similarity = [{}, {}, {}, {}, {}]
    for i in range(N_folds):
        if model_list[i].opt_method == "cos" and os.path.exists(
            model_list[i].cos_dump_path
        ):
            with open(model_list[i].cos_dump_path, "rb") as f:
                similarity[i] = pickle.load(f)
        if model_list[i].opt_method == "euc" and os.path.exists(
            model_list[i].euc_dump_path
        ):
            with open(model_list[i].euc_dump_path, "rb") as f:
                similarity[i] = pickle.load(f)
    result_file = "./results/result_" + model_list[0].opt_method + ".txt"
    with open(result_file, "w") as w_file:
        for userID, itemlist in test_data.items():
            w_file.write(str(userID) + "|" + str(itemlist[0]) + "\n")
            for i in range(itemlist[0]):
                itemID = itemlist[i + 1]
                r_ui_h = [0, 0, 0, 0, 0]
                for i in range(N_folds):
                    r_ui_h[i] = (
                        model_list[i].global_mean
                        + model_list[i].user_bias[userID]
                        + model_list[i].item_bias[itemID]
                        + np.dot(model_list[i].pu[userID], model_list[i].qi[itemID])
                    )
                    if userID in similarity[i].keys():
                        item_simi = similarity[i][userID]
                        if itemID in item_simi.keys():
                            smi_rate = 0.3
                            similarity_score = item_simi[itemID]
                            if similarity_score == 0:
                                smi_rate = 0
                            r_ui_h[i] = (
                                r_ui_h[i] * (1 - smi_rate) + similarity_score * smi_rate
                            )
                avg_score = np.mean(r_ui_h)
                avg_score = min(100, max(0, avg_score))
                w_file.write(str(itemID) + "  " + str(avg_score) + "\n")


def baiscfunksvd_avg_test(test_data, model_list):
    """
    使用基础算法对测试集进行评分
    """
    with open("./results/result.txt", "w") as w_file:
        for userID, itemlist in test_data.items():
            w_file.write(str(userID) + "|" + str(itemlist[0]) + "\n")
            for i in range(itemlist[0]):
                itemID = itemlist[i + 1]
                r_ui_h = [0, 0, 0, 0, 0]
                for i in range(N_folds):
                    r_ui_h[i] = (
                        model_list[i].global_mean
                        + model_list[i].user_bias[userID]
                        + model_list[i].item_bias[itemID]
                        + np.dot(model_list[i].pu[userID], model_list[i].qi[itemID])
                    )
                avg_score = np.mean(r_ui_h)
                avg_score = min(100, max(0, avg_score))
                w_file.write(str(itemID) + "  " + str(avg_score) + "\n")


if __name__ == "__main__":
    all_data = get_train_data(train_dump_path)  # 这个alldata指的是train.txt所有数据
    test_data = get_test_data(test_dump_path)  # 这个testdata指的是test.txt
    attr = get_attr(attr_dump_path)  # itemAttribute.txt
    # 交叉验证
    model_list = []
    for i in range(N_folds):
        model = FunkSVD(FOLD=i, K=K, optim=True)
        train_data, valid_data = cross_val_score(all_data, N_folds, i)
        # model.dump_valid_cos_simi(valid_data, train_data, i)
        # model.dump_valid_euc_simi(valid_data, train_data)
        # print("finish dump valid euc similarity, fold: ", i)
        model.train(train_data, valid_data, EPOCH=EPOCH, FOLD=i)
        print("Fold " + str(i) + " ,rmse on all data:", model.RMSE(all_data))
        # print(
        #     "Fold " + str(i) + " ,opt rmse on all data:", model.opt_RMSE(all_data, True)
        # )
        with open("./models/funkSVD_" + str(i) + ".pkl", "rb") as f:
            model = pickle.load(f)
            # model.predict(train_data, test_data)
            model_list.append(model)

    print("final opt rmse on all data:", avg_predict(all_data, model_list, True))
    baiscfunksvd_avg_test(test_data, model_list)
    # 模型聚合,下面的代码均已弃用
    # with open(model_list[0].save_path, "rb") as f:
    #     finalModel = pickle.load(f)
    #     finalModel.save_path = "./models/opt_euc_funkSVD_final.pkl"
    #     print("Fold " + str(0) + ",best rmse on all data:", finalModel.RMSE(all_data))
    #     print(
    #         "Fold " + str(0) + " ,best opt rmse on all data:",
    #         finalModel.opt_RMSE(all_data, True),
    #     )
    # for i in range(1, N_folds):
    #     with open(model_list[i].save_path, "rb") as f:
    #         model = pickle.load(f)
    #     print("Fold " + str(i) + ",best rmse on all data:", model.RMSE(all_data))
    #     print(
    #         "Fold " + str(i) + " ,best opt rmse on all data:",
    #         model.opt_RMSE(all_data, True),
    #     )
    #     finalModel.user_bias += model.user_bias
    #     finalModel.item_bias += model.item_bias
    #     finalModel.pu += model.pu
    #     finalModel.qi += model.qi
    #     finalModel.global_mean += model.global_mean
    # finalModel.user_bias /= N_folds
    # finalModel.item_bias /= N_folds
    # finalModel.pu /= N_folds
    # finalModel.qi /= N_folds
    # finalModel.global_mean /= N_folds
    # finalModel.save()
    # with open(finalModel.save_path, "rb") as f:
    #     # with open("./models/OptimFunkSVD_0.pkl", "rb") as f:
    #     model = pickle.load(f)
    #     print("final rmse on all data:", model.RMSE(all_data))
    #     print("final opt rmse on all data:", model.opt_RMSE(all_data, True))
    #     # print("best rmse", model.best_rmse)
    #     model.predict(all_data, test_data)
