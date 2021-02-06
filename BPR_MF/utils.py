"""
@Author: YMH
@Date:
@Description: 定义一些必要的工具函数，包括数据预处理、特征工程、模型的训练和预测函数等
"""

import pandas as pd
import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


def preprocess(path, test_neg_num=100):
    df = pd.read_csv(path, sep="::", engine="python", names=['user_id', 'item_id', 'rating', 'timestamp'])
    # 剔除评分次数小于5次的项目,评分小于2的样本和评分次数小于等于2次的用户
    df['item_count'] = df.groupby('item_id')['item_id'].transform('count')
    df = df[df.item_count >= 5]
    df = df[df.rating >= 2]
    df['user_count'] = df.groupby('user_id')['user_id'].transform('count')
    df = df[df.user_count > 2]
    df = df.sort_values(by=['user_id', 'timestamp'])

    user_num, item_num = df['user_id'].max() + 1, df['item_id'].max() + 1
    params_dict = {'user_num': user_num, 'item_num': item_num}
    # 进行负采样，划分训练集、验证集和测试集
    train_dict, val_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = df['item_id'].max()
    for user_id, data_df in df[['user_id', 'item_id']].groupby('user_id'):
        pos_list = data_df['item_id'].tolist()
        neg_list = []
        while len(neg_list) < len(pos_list) + test_neg_num:
            temp = random.randint(1, item_id_max)
            if temp not in set(pos_list):
                neg_list.append(temp)
        train_dict['user_id'].extend([user_id] * (len(pos_list) - 2))
        train_dict['pos_id'].extend(pos_list[:len(pos_list) - 2])
        train_dict['neg_id'].extend(neg_list[:len(pos_list) - 2])
        val_dict['user_id'].append(user_id)
        val_dict['pos_id'].append(pos_list[len(pos_list) - 2])
        val_dict['neg_id'].append(neg_list[len(pos_list) - 2])
        test_dict['user_id'].append(user_id)
        test_dict['pos_id'].append(pos_list[len(pos_list) - 1])
        test_dict['neg_id'].append(neg_list[len(pos_list) - 1:])
    train_df = pd.DataFrame(train_dict)
    val_df = pd.DataFrame(val_dict)
    train_df = train_df.sample(frac=1.).reset_index(drop=True)
    val_df = val_df.sample(frac=1.).reset_index(drop=True)
    test_data = [np.array(test_dict['user_id']), np.array(test_dict['pos_id']), np.array(test_dict['neg_id'])]

    return params_dict, train_df, val_df, test_data


class BPRDataset(Dataset):
    """
    定义BPR_MF模型测试的dataset
    """
    def __init__(self, df):
        self.user_id = df['user_id'].values
        self.pos_id = df['pos_id'].values
        self.neg_id = df['neg_id'].values

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, index):
        return self.user_id[index], self.pos_id[index], self.neg_id[index]


class BPRLoss(nn.Module):
    """
    定义BPR模型的损失函数类
    """
    def __init__(self):
        super(BPRLoss, self).__init__()

    @staticmethod
    def forward(pos_score, neg_score):
        return - F.sigmoid(pos_score - neg_score).log().sum()


# def bpr_loss(pos_score, neg_score):
#     """
#     定义BPR模型的损失函数
#     """
#     return - F.sigmoid(pos_score - neg_score).log().sum()


def train(model, epochs, loader, criterion, optimizer):
    """
    用于模型的训练
    :param model: 训练的模型
    :param epochs: 迭代次数
    :param loader: 数据加载器dataloader
    :param criterion: 损失评估函数
    :param optimizer: 优化器
    :return:
    """
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n = 0
        for i, (user_id, pos_id, neg_id) in enumerate(loader):
            optimizer.zero_grad()
            pos, neg = model(user_id, pos_id, neg_id)
            loss = criterion(pos, neg)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n += 1
        print("Epoch: {}, train_loss: {:.4f}".format(epoch + 1, train_loss / n))


def test(model, loader, criterion):
    """
    用于模型的测试
    :param model: 训练完成的模型
    :param epochs: 迭代次数
    :param loader: 数据集加载器
    :param criterion: 损失评估函数
    :return:
    """
    test_loss = 0
    n = 0
    with torch.no_grad():
        for i, (user_id, pos_id, neg_id) in enumerate(loader):
            pos, neg = model(user_id, pos_id, neg_id)
            loss = criterion(pos, neg)
            test_loss += loss.item()
            n += 1
    print("test_loss: {:.4f}".format(test_loss / n))


def evaluate(model, test_data, top_k):
    """
    用于模型评估
    :param model: 评估的模型
    :param test_data: 评估数据集
    :param top_k: 前k个项目
    :return:
    """
    hr, ndcg = [], []
    user_list, pos_list, neg_list = torch.from_numpy(test_data[0]), torch.from_numpy(test_data[1]), torch.from_numpy(test_data[2])
    for user_id, pos_id, neg_id in zip(user_list, pos_list, neg_list):
        items = torch.cat([pos_id.unsqueeze(0), neg_id])
        pos_score, neg_score = model(user_id, pos_id, neg_id)
        score = torch.cat([pos_score.unsqueeze(0), neg_score])
        _, idx = torch.topk(score, top_k)
        recommends = torch.take(items, idx).numpy().tolist()
        if pos_id in recommends:
            hr.append(1)
            index = recommends.index(pos_id)
            ndcg.append(1 / np.log2(index + 2))
        else:
            hr.append(0)
            ndcg.append(0)
    return np.mean(hr), np.mean(ndcg)


if __name__ == "__main__":
    # preprocess()
    # A = defaultdict(list)
    # A['a'] = [1, 2, 3]
    # A['b'] = [1, 2, 3]
    # A['c'] = [1, 2, 3]
    # print(A)
    # print(A.values())
    # random.shuffle(A.keys())
    # print(A)
    # print(A.values())
    a = torch.ones(3)
    print(a)
    print(a.sum(dim=0))
    print(a.sum(dim=-1))
    print(a.sum(dim=1))
