"""
@Author: YMH
@Date: 2021-1-31
@Description: 基于MovieLens 1M movie ratings数据集，下载地址：https://grouplens.org/datasets/movielens/
"""

import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from MF import MF
from utils import *

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # =============== 初始化命令行参数 ===============
    parser = argparse.ArgumentParser(description="MF model test sample on MovieLens 1M movie ratings dataset")
    parser.add_argument("--file_path", type=str, default="../dataset/ml-1m/ratings.dat", help="数据集路径")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率")
    parser.add_argument("--batch_size", type=int, default=512, help="批量大小")
    parser.add_argument("--epochs", type=int, default=10, help="迭代次数")
    args = parser.parse_args()

    # =============== 参数设置 ===============
    hidden_dim = 32
    test_size = 0.2

    # =============== 创建训练集和测试集 ===============
    params_dict, train_df, test_df = preprocess(args.file_path, test_size)
    user_num, item_num = params_dict['user_num'], params_dict['item_num']

    # =============== 创建模型 ===============
    mf_model = MF(user_num, item_num, hidden_dim)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(mf_model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

    # =============== 构建数据集加载器 ===============
    train_dataset = MFDataset(train_df)
    test_dataset = MFDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # =============== 模型训练与测试 ===============
    train(mf_model, args.epochs, train_loader, loss_func, optimizer)
    test(mf_model, args.epochs, train_loader, loss_func)
