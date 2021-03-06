"""
@Author: YMH
@Date: 2021-4-13
@Description: 基于Criteo数据集测试xDeepFM模型，数据集下载地址：https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310
"""

import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from xDeepFM import xDeepFM
from utils import *

if __name__ == "__main__":
    # =============== 初始化命令行参数 ===============
    parser = argparse.ArgumentParser(description="xDeepFM model test sample on Criteo dataset")
    parser.add_argument("--file_path", type=str, default="../dataset/Criteo/train.txt", help="数据集读取路径")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=2048, help="批量大小")
    parser.add_argument("--epochs", type=int, default=10, help="迭代次数")
    args = parser.parse_args()

    # =============== 参数设置 ===============
    sample_num = 300000  # 取部分数据进行测试
    test_size = 0.2
    k = 8
    dropout = 0.5
    output_dim = 1
    split_half = True
    reg = 1e-4

    # =============== 准备数据 ===============
    dense_feature = ['I' + str(i) for i in range(1, 14)]
    sparse_feature = ['C' + str(i) for i in range(1, 27)]
    embed_dict, train_df, test_df = preprocess(args.file_path, sample_num, test_size)
    embed_num = list(embed_dict.values())
    cin_layers = [128, 128]
    concat_dim = len(dense_feature) + len(sparse_feature) * k
    hidden_units = [concat_dim, 256, 128, 64]
    train_dataset = XDeepFMDataset(train_df, dense_feature, sparse_feature)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # =============== 创建模型 ===============
    xDeepFM_model = xDeepFM(embed_num, k, cin_layers, hidden_units, dropout, output_dim, split_half)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(xDeepFM_model.parameters(), lr=args.learning_rate, weight_decay=reg)

    # =============== 模型训练与测试 ===============
    train(xDeepFM_model, args.epochs, train_loader, loss_func, optimizer)
    test(xDeepFM_model, test_df, dense_feature, sparse_feature)
