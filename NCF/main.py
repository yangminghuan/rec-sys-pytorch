"""
@Author: YMH
@Date: 2021-2-27
@Description: 基于MovieLens 1M movie ratings数据集，下载地址：https://grouplens.org/datasets/movielens/
"""

import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from NCF import NCF
from utils import *


if __name__ == "__main__":
    # =============== 初始化命令行参数 ===============
    parser = argparse.ArgumentParser(description="NCF model test sample on MovieLens 1M movie ratings dataset")
    parser.add_argument("--file_path", type=str, default="../dataset/ml-1m/ratings.dat", help="数据集路径")
    parser.add_argument("--test_neg_num", type=int, default=100, help="测试集负样本数量")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=256, help="批量大小")
    parser.add_argument("--epochs", type=int, default=20, help="迭代次数")
    args = parser.parse_args()

    # =============== 参数设置 ===============
    embed_dim = 32
    layers = [64, 32, 16, 8]
    # reg = 1e-6
    top_k = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # =============== 准备数据 ===============
    params_dict, train_df, val_df, test_data = preprocess(args.file_path, args.test_neg_num)
    user_num, item_num = params_dict['user_num'], params_dict['item_num']
    train_dataset = NCFDataset(train_df)
    val_dataset = NCFDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # =============== 创建模型 ===============
    ncf_model = NCF(user_num, item_num, embed_dim, layers)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(ncf_model.parameters(), lr=args.learning_rate)

    # =============== 模型训练与测试 ===============
    train(ncf_model, args.epochs, train_loader, loss_func, optimizer)
    test(ncf_model, val_loader, loss_func)

    # =============== 模型评估 ===============
    HR, NDCG = evaluate(ncf_model, test_data, top_k)
    print("HR: ", HR, "NDCG: ", NDCG)
