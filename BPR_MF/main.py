"""
@Author: YMH
@Date:
@Description: 基于MovieLens 1M movie ratings数据集，下载地址：https://grouplens.org/datasets/movielens/
"""

import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from BPR import BPR
from utils import *

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # =============== 初始化命令行参数 ===============
    parser = argparse.ArgumentParser(description="BPR model test sample on MovieLens 1M movie ratings dataset")
    parser.add_argument("--file_path", type=str, default="../dataset/ml-1m/ratings.dat", help="数据集路径")
    parser.add_argument("--test_neg_num", type=int, default=100, help="测试集负样本数量")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=512, help="批量大小")
    parser.add_argument("--epochs", type=int, default=20, help="迭代次数")
    args = parser.parse_args()

    # =============== 参数设置 ===============
    embed_dim = 32
    reg = 1e-6
    top_k = 10

    # =============== 准备数据 ===============
    params_dict, train_df, val_df, test_data = preprocess(args.file_path, args.test_neg_num)
    user_num, item_num = params_dict['user_num'], params_dict['item_num']
    train_dataset = BPRDataset(train_df)
    val_dataset = BPRDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # =============== 创建模型 ===============
    bpr_model = BPR(user_num, item_num, embed_dim)
    loss_func = BPRLoss()
    optimizer = optim.Adam(bpr_model.parameters(), lr=args.learning_rate, weight_decay=reg)

    # =============== 模型训练与测试 ===============
    train(bpr_model, args.epochs, train_loader, loss_func, optimizer)
    test(bpr_model, val_loader, loss_func)

    # =============== 模型评估 ===============
    HR, NDCG = evaluate(bpr_model, test_data, top_k)
    print("HR: ", HR, "NDCG: ", NDCG)
