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

EPOCH = 3
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


def cross_val_score( all_data, n_folds, fold):
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


if __name__ == "__main__":
    all_data = get_train_data(train_dump_path)  # 这个alldata指的是train.txt所有数据
    test_data = get_test_data(test_dump_path)  # 这个testdata指的是test.txt
    attr = get_attr(attr_dump_path)  # itemAttribute.txt
    # 交叉验证
    model_list = []
    for i in range(N_folds):
        model = FunkSVD(FOLD=i, K=K, optim=True)
        train_data, valid_data = cross_val_score(all_data, N_folds, i)
        # model.dump_valid_cos_simi(valid_data,train_data)
        # print("finish dump valid cos similarity, fold: ",i)
        model.train(train_data, valid_data, EPOCH=EPOCH, FOLD=i)
        print("Fold "+str(i)+" ,rmse on all data:", model.RMSE(all_data))
        print("Fold "+str(i)+" ,opt rmse on all data:", model.opt_RMSE(all_data, True))
        model_list.append(model)
   #模型聚合
    finalModel = FunkSVD(FOLD='r', K=K, optim=True)
    for i in range(0, N_folds):
        with open(model_list[i].save_path, "rb") as f:
            model = pickle.load(f)
        finalModel.user_bias += model_list[i].user_bias
        finalModel.item_bias += model_list[i].item_bias
        finalModel.pu += model_list[i].pu
        finalModel.qi += model_list[i].qi
        finalModel.global_mean += model_list[i].global_mean
    finalModel.user_bias /= N_folds
    finalModel.item_bias /= N_folds
    finalModel.pu /= N_folds
    finalModel.qi /= N_folds
    finalModel.global_mean /= N_folds
    finalModel.save()
    with open(finalModel.save_path, "rb") as f:
    # with open("./models/OptimFunkSVD_0.pkl", "rb") as f:
        model = pickle.load(f)
        print("final rmse on all data:", model.RMSE(all_data))
        print("final opt rmse on all data:", model.opt_RMSE(all_data,True))
        # print("best rmse", model.best_rmse)
        model.predict(all_data, test_data)
