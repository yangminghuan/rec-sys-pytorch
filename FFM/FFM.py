"""
@Author: YMH
@Date: 2021-2-21
@Description: Field-aware Factorization Machines模型代码复现
"""

import torch
import torch.nn as nn


class FFM(nn.Module):
    """
    域感知因子分解机推荐模型
    """
    def __init__(self, feature_size, field_size, k):
        """
        :param feature_size: 特征数量
        :param field_size: field大小
        :param k: 辅助向量的维度大小
        """
        super(FFM, self).__init__()
        # 初始化参数
        self.feature_size = feature_size
        self.field_size = field_size
        self.k = k

        # 定义模型层
        self.linear = nn.Linear(feature_size, 1)
        self.v = nn.Parameter(torch.normal(0, 0.01, size=(feature_size, field_size, k)))

    def forward(self, x):
        # 线性部分
        linear_part = self.linear(x).squeeze()

        # 特征交叉部分
        field_x = torch.tensordot(x, self.v, dims=1)
        interaction = 0
        for i in range(self.field_size):
            for j in range(i+1, self.field_size):
                field_xij = field_x[:, i] * field_x[:, j]
                interaction += torch.sum(field_xij, dim=1)

        result = torch.sigmoid(linear_part + interaction)
        return result
