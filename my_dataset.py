import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.user_id = torch.LongTensor(df["userID"].to_list())
        self.item_id = torch.LongTensor(df["itemID"].to_list())
        self.score = torch.Tensor(df["score"].to_list())

    def __getitem__(self, idx):
        return self.user_id[idx], self.item_id[idx], self.score[idx]

    def __len__(self):
        return self.score.shape[0]
