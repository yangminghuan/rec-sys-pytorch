"""
@Author: YMH
@Date: 2021-3-13
@Description: Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features论文模型代码复现
"""

import torch
import torch.nn as nn


class EmbedLayer(nn.Module):
    """
    Embedding层：将稀疏的类别型特征转换成稠密的Embedding向量
    """
    def __init__(self, embed_num, embed_dim):
        """
        :param embed_num: 稀疏特征嵌入数量
        :param embed_dim: 嵌入向量维度大小
        """
        super(EmbedLayer, self).__init__()
        # 初始化参数
        self.embed_num = embed_num
        self.embed_dim = embed_dim

        # 定义网络层
        self.embed_layers = nn.ModuleList()
        for num in embed_num:
            self.embed_layers.append(nn.Embedding(num, embed_dim))

        # 初始化网络层
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        for embed in self.embed_layers:
            nn.init.uniform_(embed.weight, -0.05, 0.05)

    def forward(self, sparse_input):
        sparse_embed = torch.cat([self.embed_layers[i](sparse_input[:, i])
                                  for i in range(len(self.embed_num))], dim=-1)
        return sparse_embed


class ResidualUnit(nn.Module):
    """
    Residual Unit层：对特征向量各个维度进行充分的交叉组合，抓取更多的非线性特征和组合特征的信息
    """
    def __init__(self, hidden_dim, stack_dim):
        """
        :param hidden_dim: 隐藏层单元维度
        :param stack_dim: 堆叠层维度
        """
        super(ResidualUnit, self).__init__()
        # 初始化参数
        self.hidden_units = hidden_dim
        self.stack_dim = stack_dim

        # 定义网络层
        self.fc1 = nn.Linear(stack_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, stack_dim)

        # 初始化网络层参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.zero_()

    def forward(self, stack_input):
        x = self.fc1(stack_input)
        x = torch.relu(x)
        x = self.fc2(x)
        output = torch.relu(x + stack_input)
        return output


class DeepCrossing(nn.Module):
    """
    Deep Crossing模型
    """
    def __init__(self, embed_num, embed_dim, hidden_units, stack_dim):
        """
        :param embed_num: 稀疏特征嵌入数量
        :param embed_dim: 嵌入向量维度大小
        :param hidden_units: 隐藏层单元维度
        :param stack_dim: 堆叠层维度
        """
        super(DeepCrossing, self).__init__()
        # 初始化参数
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.stack_dim = stack_dim

        # 定义网络层
        self.embed_layer = EmbedLayer(embed_num, embed_dim)
        self.res_layers = nn.ModuleList()
        for hidden_dim in hidden_units:
            self.res_layers.append(ResidualUnit(hidden_dim, stack_dim))
        self.score_layer = nn.Linear(stack_dim, 1)

        # 初始化网络参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        nn.init.xavier_uniform_(self.score_layer.weight)
        self.score_layer.bias.data.zero_()

    def forward(self, dense_input, sparse_input):
        """
        :param dense_input: 数值型特征输入
        :param sparse_input: 稀疏类别型特征输入
        :return: CTR点击率预估
        """
        # Embedding层
        sparse_embed = self.embed_layer(sparse_input)

        # Stacking层
        stack_input = torch.cat([dense_input, sparse_embed], dim=-1)

        # Multiple Residual Units层
        score_input = stack_input
        for res in self.res_layers:
            score_input = res(score_input)

        # Scoring层
        result = torch.sigmoid(self.score_layer(score_input)).squeeze()

        return result
