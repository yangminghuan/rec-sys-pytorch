"""
@Author: YMH
@Date: 2021-1-31
@Description: 定义一些必要的工具函数，包括数据预处理、特征工程、模型的训练和预测函数等
"""

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


def preprocess(path, test_size=0.2):
    """
    :param path: 数据集路径
    :param test_size: 测试集大小
    :return: 一些必要的参数，训练集和测试集
    """
    # 读取数据集
    df = pd.read_csv(path, sep="::", engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    df['avg_score'] = df.groupby('userId')['rating'].transform('mean')
    user_num, item_num = df['userId'].max() + 1, df['movieId'].max() + 1
    params_dict = {'user_num': user_num, 'item_num': item_num}

    # 划分训练集和测试集
    score_count = df.groupby('userId')['movieId'].agg('count')
    test_df = pd.DataFrame()
    for i in score_count.index:
        temp = df[df.userId == i].iloc[int((1 - test_size) * score_count[i]):]
        test_df = pd.concat([test_df, temp], axis=0)
    test_df.reset_index(inplace=True)
    train_df = df.drop(index=test_df['index'])
    train_df = train_df.drop(['timestamp'], axis=1)
    train_df = train_df.sample(frac=1.).reset_index(drop=True)  # 随机打散样本数据，重新排列
    test_df = test_df.drop(['index', 'timestamp'], axis=1)
    test_df = test_df.sample(frac=1.).reset_index(drop=True)  # 随机打散样本数据，重新排列

    return params_dict, train_df, test_df


class MFDataset(Dataset):
    """
    定义MF模型测试的dataset
    """
    def __init__(self, df):
        self.avg_score = df['avg_score'].values
        self.user_id = df['userId'].values
        self.item_id = df['movieId'].values
        self.label = df['rating'].values.astype('float64')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.avg_score[index], self.user_id[index], self.item_id[index], self.label[index]


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
        for i, (avg_score, user_id, item_id, label) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(user_id, item_id, avg_score)
            loss = criterion(pred, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n += 1
        print("Epoch: {}, train_loss: {:.4f}".format(epoch + 1, train_loss / n))


def test(model, epochs, loader, criterion):
    """
    用于模型的测试
    :param model: 训练完成的模型
    :param epochs: 迭代次数
    :param loader: 数据集加载器
    :param criterion: 损失评估函数
    :return:
    """
    for epoch in range(epochs):
        model.eval()
        test_loss = 0
        n = 0
        with torch.no_grad():
            for i, (avg_score, user_id, item_id, label) in enumerate(loader):
                pred = model(user_id, item_id, avg_score)
                loss = criterion(pred, label)
                test_loss += loss.item()
                n += 1
        print("Epoch: {}, test_loss: {:.4f}".format(epoch + 1, test_loss / n))
