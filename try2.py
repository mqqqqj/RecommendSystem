import os
import math
import pickle
import numpy as np
from funksvd import FunkSVD


train_path = "./data/train.txt"

test_path = "./data/test.txt"

dump_path = "./data/train.pkl"

EPOCH = 10
K = 100


def dump_data(src_path, dumped_path):
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
        data = dump_data(train_path, dumped_path)
        return data


if __name__ == "__main__":
    train_data = get_train_data(dump_path)
    # print(train_data[15143][338188])
    model = FunkSVD(K=K)
    model.train(train_data, EPOCH=EPOCH)
