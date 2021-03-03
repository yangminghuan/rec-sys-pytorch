"""
@Author: YMH
@Date: 2021-3-3
@Description: 基于Criteo数据集测试Wide & Deep模型，数据集下载地址：https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310
"""

import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from WDL import WDL
from utils import *

if __name__ == "__main__":
    # =============== 初始化命令行参数 ===============
    parser = argparse.ArgumentParser(description="WDL model test sample on Criteo dataset")
    parser.add_argument("--file_path", type=str, default="../dataset/Criteo/train.txt", help="数据集读取路径")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率")
    parser.add_argument("--batch_size", type=int, default=256, help="批量大小")
    parser.add_argument("--epochs", type=int, default=20, help="迭代次数")
    args = parser.parse_args()

    # =============== 参数设置 ===============
    sample_num = 20000  # 取部分数据进行测试
    test_size = 0.2
    k = 8
    reg = 1e-4

    # =============== 准备数据 ===============
    embed_dict, train_df, test_df = preprocess(args.file_path, sample_num, test_size)
    embed_num = list(embed_dict.values())
    sparse_feature = list(embed_dict.keys())
    layers = [k * len(embed_num), 256, 128, 64]
    output_dim = 1
    train_dataset = WDLDataset(train_df, sparse_feature)
    # test_dataset = FMDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # =============== 创建模型 ===============
    dense_size = train_df.shape[-1] - 1 - len(sparse_feature)
    wdl_model = WDL(embed_num, k, dense_size, layers, output_dim)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(wdl_model.parameters(), lr=args.learning_rate, weight_decay=reg)

    # =============== 模型训练与测试 ===============
    train(wdl_model, args.epochs, train_loader, loss_func, optimizer)
    test(wdl_model, test_df, sparse_feature)
