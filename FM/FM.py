"""
@Author: YMH
@Date:
@Description: Factorization Machines模型代码复现
"""

import torch
import torch.nn as nn


class FM(nn.Module):
    """
    因子分解机推荐模型
    """
    def __init__(self, feature_size, k):
        """
        :param feature_size: 特征数量
        :param k: 辅助向量的维度大小
        """
        super(FM, self).__init__()
        # 初始化参数
        self.feature_size = feature_size
        self.k = k

        # 定义模型层
        self.linear = nn.Linear(feature_size, 1)
        self.v = nn.Parameter(torch.normal(0, 0.01, size=(feature_size, k)))

    def forward(self, x):
        # 线性部分
        linear_part = self.linear(x).squeeze()

        # 特征交叉部分
        part1 = torch.sum(torch.pow(torch.matmul(x, self.v), 2), dim=1)
        x_square, v_square = torch.pow(x, 2), torch.pow(self.v, 2)
        part2 = torch.sum(torch.matmul(x_square, v_square), dim=1)
        interaction = 0.5 * (part1 - part2)

        result = torch.sigmoid(linear_part + interaction)
        return result
