"""
@Author: YMH
@Date: 2021-2-14
@Description: 基于Criteo数据集测试FM因子分解机模型，数据集下载地址：https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310
"""

import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from FM import FM
from utils import *

if __name__ == "__main__":
    # =============== 初始化命令行参数 ===============
    parser = argparse.ArgumentParser(description="FM model test sample on Criteo dataset")
    parser.add_argument("--file_path", type=str, default="../dataset/Criteo/train.txt", help="数据集读取路径")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率")
    parser.add_argument("--batch_size", type=int, default=512, help="批量大小")
    parser.add_argument("--epochs", type=int, default=20, help="迭代次数")
    args = parser.parse_args()

    # =============== 参数设置 ===============
    sample_num = 20000  # 取部分数据进行测试
    test_size = 0.2
    k = 10
    reg = 1e-4

    # =============== 准备数据 ===============
    train_df, test_df = preprocess(args.file_path, sample_num, test_size)
    train_dataset = FMDataset(train_df)
    # test_dataset = FMDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # =============== 创建模型 ===============
    feature_size = train_df.shape[-1] - 1
    fm_model = FM(feature_size, k)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(fm_model.parameters(), lr=args.learning_rate, weight_decay=reg)

    # =============== 模型训练与测试 ===============
    train(fm_model, args.epochs, train_loader, loss_func, optimizer)
    test(fm_model, test_df)
