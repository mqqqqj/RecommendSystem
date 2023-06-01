def summary_user():
    users = set()
    with open("./data/train.txt", "r") as r_file:
        line = r_file.readline()
        while line:
            user_id, n_item = line.split("|")
            users.add(int(user_id))
            for _ in range(int(n_item)):
                r_file.readline().split("  ")
            line = r_file.readline()
    print("number of users is :", len(users))
    return users


def summary_items():
    items = set()
    with open("./data/itemAttribute.txt", "r") as r_file:
        line = r_file.readline()
        while line:
            i1, i2, i3 = line.split("|")
            # print(i1, i2, i3)
            items.add(int(i1))
            if i2 != "None":
                items.add(int(i2))
            if i3 != "None\n":
                items.add(int(i3))
            line = r_file.readline()
    print("number of items is :", len(items))
    return items


def summary_matrix():
    non_zero = 0
    n_users = 19835
    n_items = 624960
    with open("./data/train.txt", "r") as r_file:
        line = r_file.readline()
        while line:
            user_id, n_item = line.split("|")
            n_item = int(n_item)
            non_zero += n_item
            for _ in range(n_item):
                r_file.readline()
            line = r_file.readline()
    fill_rate = non_zero / (n_users * n_items)
    print("non zero terms is:", non_zero)
    print("matrix fill rate:", fill_rate)
    return non_zero, fill_rate


def summary_train():
    score_dict = {}
    with open("./data/train.txt", "r") as r_file:
        line = r_file.readline()
        while line:
            user_id, n_item = line.split("|")
            n_item = int(n_item)
            for _ in range(n_item):
                item_id, score = r_file.readline().split("  ")
                if int(score) not in score_dict.keys():
                    score_dict[int(score)] = 0
                else:
                    score_dict[int(score)] += 1
            line = r_file.readline()
    # print("score distribute:", score_dict)
    mean_score = 0
    vals = 0
    for k, v in score_dict.items():
        mean_score += k * v
        vals += v
    mean_score /= vals
    print("mean score:", mean_score)
    # 按value升序排列
    sorted_list = sorted(score_dict.items(), key=lambda x: x[1])
    print("最不常打出的分数", sorted_list[:4])
    print("最常打出的分数", sorted_list[-5:-1])
    return score_dict


if __name__ == "__main__":
    # 统计用户个数
    # users = summary_user()
    # 统计商品个数
    # 发现这样算item个数没用,item在train,test,attr文件里都有,
    # 且有的item只出现在其中一个文件,直接看attr文件最后一行是多少算了.
    # items = summary_items()   #废弃
    # 个数应该是这个结果:
    # n_users = 19835
    # n_items = 624960
    # 统计矩阵的填充率
    # summary_matrix()
    # 统计用户打分分数的分布
    summary_train()
