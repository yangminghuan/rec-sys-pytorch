"""
@Author: YMH
@Date: 2021-4-10
@Description: 定义一些必要的工具函数，包括数据预处理、特征工程、模型的训练和预测函数等
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def preprocess(path, sample_num, test_size=0.2):
    """
    :param path: 数据集读取路径
    :param sample_num: 实验用的数据集大小
    :param test_size: 测试集大小
    :return: 返回稀疏特征嵌入字典、训练集和测试集
    """
    # 读取数据集
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    names = ['label'] + dense_features + sparse_features
    df = pd.read_csv(path, sep='\t', header=None, nrows=sample_num, names=names)

    # 缺失值填充
    df[dense_features] = df[dense_features].fillna(0)
    df[sparse_features] = df[sparse_features].fillna('null')

    # 归一化、数值编码
    mms = MinMaxScaler()
    df[dense_features] = mms.fit_transform(df[dense_features])
    le = LabelEncoder()
    for feat in sparse_features:
        df[feat] = le.fit_transform(df[feat])

    # 计算稀疏特征的嵌入数量
    embed_dict = {}
    for feat in sparse_features:
        embed_dict[feat] = df[feat].nunique()

    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=True)

    return embed_dict, train_df, test_df


class DeepFMDataset(Dataset):
    """
    定义DeepFM模型测试用的dataset
    """
    def __init__(self, df, dense_feature, sparse_feature):
        self.dense_input = df[dense_feature].values.astype('float32')
        self.sparse_input = df[sparse_feature].values
        self.y = df['label'].values.astype('float32')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.dense_input[index], self.sparse_input[index], self.y[index]


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
        for i, (dense, sparse, y) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(dense, sparse)
            loss = criterion(pred, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n += 1
        print("Epoch: {}, train_loss: {:.4f}".format(epoch + 1, train_loss / n))


def test(model, test_df, dense_feature, sparse_feature):
    """
    用于模型的测试
    :param model: 训练完成的模型
    :param test_df: 测试数据
    :param dense_feature: 数值型特征
    :param sparse_feature: 稀疏特征
    :return:
    """
    with torch.no_grad():
        test_dense = torch.from_numpy(test_df[dense_feature].values.astype('float32'))
        test_sparse = torch.from_numpy(test_df[sparse_feature].values)
        test_y = test_df['label'].values
        pred = model(test_dense, test_sparse).numpy()
        pred = [1 if x > 0.5 else 0 for x in pred]
        acc = accuracy_score(test_y, pred)
        auc = roc_auc_score(test_y, pred)
    print("test accuracy_score: ", acc)
    print("test roc_auc_score: ", auc)
