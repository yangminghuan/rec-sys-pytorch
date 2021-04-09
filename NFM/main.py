"""
@Author: YMH
@Date: 2021-4-9
@Description: 基于Criteo数据集测试NFM神经因子分解机模型，数据集下载地址：https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310
"""

import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from NFM import NFM
from utils import *

if __name__ == "__main__":
    # =============== 初始化命令行参数 ===============
    parser = argparse.ArgumentParser(description="Neural Factorization Machines model test sample on Criteo dataset")
    parser.add_argument("--file_path", type=str, default="../dataset/Criteo/train.txt", help="数据集读取路径")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=1024, help="批量大小")
    parser.add_argument("--epochs", type=int, default=15, help="迭代次数")
    args = parser.parse_args()

    # =============== 参数设置 ===============
    sample_num = 200000  # 取部分数据进行测试
    test_size = 0.2
    k = 8
    dropout = 0.5
    reg = 1e-4

    # =============== 准备数据 ===============
    dense_feature = ['I' + str(i) for i in range(1, 14)]
    sparse_feature = ['C' + str(i) for i in range(1, 27)]
    embed_dict, train_df, test_df = preprocess(args.file_path, sample_num, test_size)
    embed_num = list(embed_dict.values())
    dense_dim = len(dense_feature)
    hidden_units = [dense_dim + k, 256, 128, 64]
    train_dataset = NFMDataset(train_df, dense_feature, sparse_feature)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # =============== 创建模型 ===============
    NFM_model = NFM(embed_num, k, dense_dim, hidden_units, dropout)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(NFM_model.parameters(), lr=args.learning_rate, weight_decay=reg)

    # =============== 模型训练与测试 ===============
    train(NFM_model, args.epochs, train_loader, loss_func, optimizer)
    test(NFM_model, test_df, dense_feature, sparse_feature)
