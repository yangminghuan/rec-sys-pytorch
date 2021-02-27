"""
@Author: YMH
@Date: 2021-2-27
@Description: 定义一些必要的工具函数，包括数据预处理、特征工程、模型的训练和预测函数等
"""

import pandas as pd
import numpy as np
import random
from collections import defaultdict
import torch
from torch.utils.data.dataset import Dataset


def preprocess(path, test_neg_num=100):
    """
    :param path: 数据读取路径
    :param test_neg_num: 测试集负采样数
    :return: 必要参数、训练集、验证集和测试集
    """
    # 读取数据
    df = pd.read_csv(path, sep="::", engine="python", names=['user_id', 'item_id', 'rating', 'timestamp'])

    # 剔除评分次数小于5次的项目,评分小于3的样本和评分次数小于等于2次的用户
    df['item_count'] = df.groupby('item_id')['item_id'].transform('count')
    df = df[df.item_count >= 5]
    df = df[df.rating >= 3]
    df['user_count'] = df.groupby('user_id')['user_id'].transform('count')
    df = df[df.user_count > 2]
    df = df.sort_values(by=['user_id'])

    # 统计用户数和项目数
    user_num, item_num = df['user_id'].max() + 1, df['item_id'].max() + 1
    params_dict = {'user_num': user_num, 'item_num': item_num}

    # 进行负采样，划分训练集、验证集和测试集
    train_dict, val_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = df['item_id'].max()
    for user_id, data_df in df[['user_id', 'item_id']].groupby('user_id'):
        pos_list = data_df['item_id'].tolist()
        length = len(pos_list)
        neg_list = []
        while len(neg_list) < length + test_neg_num:
            temp = random.randint(1, item_id_max)
            if temp not in set(pos_list):
                neg_list.append(temp)
        train_dict['user_id'].extend([user_id] * (2 * (length - 2)))
        train_dict['item_id'].extend(pos_list[:length - 2] + neg_list[:length - 2])
        train_dict['label'].extend([1] * (length - 2) + [0] * (length - 2))
        val_dict['user_id'].extend([user_id, user_id])
        val_dict['item_id'].extend([pos_list[length - 2], neg_list[length - 2]])
        val_dict['label'].extend([1, 0])
        test_dict['user_id'].append([user_id] * (test_neg_num + 2))
        test_dict['pos_id'].append(pos_list[length - 1])
        test_dict['neg_id'].append(neg_list[length - 1:])
    train_df = pd.DataFrame(train_dict)
    val_df = pd.DataFrame(val_dict)
    test_data = [np.array(test_dict['user_id']), np.array(test_dict['pos_id']), np.array(test_dict['neg_id'])]

    return params_dict, train_df, val_df, test_data


class NCFDataset(Dataset):
    """
    定义NCF模型测试的dataset
    """
    def __init__(self, df):
        self.user_id = df['user_id'].values
        self.item_id = df['item_id'].values
        self.label = df['label'].values

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.user_id[index], self.item_id[index], self.label[index]


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
        for i, (user_id, item_id, label) in enumerate(loader):
            label = label.float()
            optimizer.zero_grad()
            pred = model(user_id, item_id)
            loss = criterion(pred, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n += 1
        print("Epoch: {}, train_loss: {:.4f}".format(epoch + 1, train_loss / n))


def test(model, loader, criterion):
    """
    用于模型的测试
    :param model: 训练完成的模型
    :param loader: 数据集加载器
    :param criterion: 损失评估函数
    :return:
    """
    test_loss = 0
    n = 0
    with torch.no_grad():
        for i, (user_id, item_id, label) in enumerate(loader):
            label = label.float()
            pred = model(user_id, item_id)
            loss = criterion(pred, label)
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
    user_list = torch.from_numpy(test_data[0])
    pos_list = torch.from_numpy(test_data[1])
    neg_list = torch.from_numpy(test_data[2])
    for user_id, pos_id, neg_id in zip(user_list, pos_list, neg_list):
        items = torch.cat([pos_id.unsqueeze(0), neg_id])
        score = model(user_id, items)
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
