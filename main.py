import os
import copy
import math
import pickle
import numpy as np
from funksvd import FunkSVD

train_path = "./data/train.txt"

test_path = "./data/test.txt"

train_dump_path = "./data/train.pkl"

test_dump_path = "./data/test.pkl"

EPOCH = 4
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
            data[int(user_id)] = []
            for _ in range(int(n_item)):
                item_id = r_file.readline().decode()
                data[int(user_id)].append(item_id)
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


def cross_val_score(model, all_data, n_folds, fold):
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

    model.train(train_data, valid_data, EPOCH=EPOCH, FOLD=fold)


if __name__ == "__main__":
    # all_data = get_train_data(train_dump_path)
    # model = FunkSVD(K=K)

    # for i in range(N_folds):
    #     cross_val_score(model, all_data, N_folds, i)
    # with open(model.save_path, "rb") as f:
    #     model = pickle.load(f)
    # print(model.RMSE(all_data))
    # print(model.best_rmse)
    test_data = get_test_data(test_dump_path)

    with open("./models/funkSVD-4epoch.pkl", "rb") as f:
        model = pickle.load(f)
