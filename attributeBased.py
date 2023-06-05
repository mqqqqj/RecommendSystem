# 我们对商品进行余弦相似度计算
import numpy as np

K = 5


def cosine_similarity(item_attribute, item_a, item_b):
    attr_a = item_attribute[item_a]
    attr_b = item_attribute[item_b]
    mo_a = np.sqrt(np.square(attr_a[0]) + np.square(attr_a[1]))
    mo_b = np.sqrt(np.square(attr_b[0]) + np.square(attr_b[1]))
    res = np.dot(attr_a, attr_b) / (mo_a * mo_b)
    return res


def get_score(item_attribute, train_data, userID, itemID):
    items = train_data[userID]
    similarity_dict = dict()
    for item in items.keys():
        cos = cosine_similarity(item_attribute, itemID, item)
        similarity_dict[item] = cos
    sorted_list = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=False)
    score = 0
    for i in range(K):
        score += train_data[userID][sorted_list[i]]
    # score /= K
    return score
