"""
@Author: YMH
@Date: 2021-3-3
@Description: Wide & Deep Learning for Recommender Systems论文模型代码复现
"""

import torch
import torch.nn as nn


class WideLayer(nn.Module):
    """Wide部分"""
    def __init__(self, dense_size):
        """
        :param dense_size: dense_feature数量
        """
        super(WideLayer, self).__init__()
        # 初始化参数
        self.dense_size = dense_size

        # 定义模型层
        self.linear = nn.Linear(dense_size, 1)

        # 初始化模型参数
        self._init_weight()

    def _init_weight(self):
        """初始化模型参数"""
        nn.init.normal_(self.linear.weight, 0, 0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x)
        return x


class DeepLayer(nn.Module):
    """Deep部分"""
    def __init__(self, layers, output_dim):
        """
        :param layers: Deep层维度大小
        :param output_dim: Deep层输出维度
        """
        super(DeepLayer, self).__init__()
        # 初始化参数
        self.layers = layers
        self.output_dim = output_dim

        # 定义网络层
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())
        self.output = nn.Linear(layers[-1], output_dim)

        # 初始化网络层
        self._init_weight()

    def _init_weight(self):
        """初始化网络参数"""
        for fc in self.fc_layers:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_normal_(fc.weight)
            if isinstance(fc, nn.Linear) and fc.bias is not None:
                fc.bias.data.zero_()
        nn.init.xavier_normal_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, x):
        for fc in self.fc_layers:
            x = fc(x)
        output = self.output(x)
        return output


class WDL(nn.Module):
    """
    Wide & Deep Learning
    """
    def __init__(self, embed_num, embed_dim, dense_size, layers, output_dim):
        """
        :param embed_num: 稀疏特征嵌入数量
        :param embed_dim: 嵌入向量维度
        :param dense_size: dense_feature数量
        :param layers: Deep层维度大小
        :param output_dim: Deep层输出维度
        """
        super(WDL, self).__init__()
        # 初始化参数
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.dense_size = dense_size
        self.layers = layers
        self.output_dim = output_dim

        # 定义网络层
        self.embed_layers = nn.ModuleList()
        for num in embed_num:
            self.embed_layers.append(nn.Embedding(num, embed_dim))
        self.wide_layer = WideLayer(dense_size)
        self.deep_layer = DeepLayer(layers, output_dim)

        # 初始化网络层
        self._init_weight()

    def _init_weight(self):
        """初始化网络参数"""
        for embed in self.embed_layers:
            nn.init.uniform_(embed.weight, -0.05, 0.05)

    def forward(self, dense_input, sparse_input):
        """
        :param dense_input: Wide层连续特征输入
        :param sparse_input: Deep层稀疏特征输入
        :return: CTR点击率预估
        """
        # Wide部分
        wide_output = self.wide_layer(dense_input)

        # Deep部分
        sparse_embed = torch.cat([self.embed_layers[i](sparse_input[:, i])
                                  for i in range(len(self.embed_num))], dim=-1)
        deep_output = self.deep_layer(sparse_embed)

        result = torch.sigmoid(0.5 * (wide_output + deep_output)).squeeze()
        return result
