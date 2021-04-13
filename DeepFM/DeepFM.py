"""
@Author: YMH
@Date: 2021-4-10
@Description: A Factorization-Machine based Neural Network for CTR Prediction论文模型代码复现
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
        sparse_embed = torch.stack([self.embed_layers[i](sparse_input[:, i])
                                    for i in range(len(self.embed_num))], dim=1)
        return sparse_embed


class FMLayer(nn.Module):
    """
    FM层包括线性部分和交叉部分，加强浅层网络部分特征组合的能力
    """
    def __init__(self, concat_dim):
        """
        :param concat_dim: 连续特征和embedding后的稀疏特征拼接后的维度大小
        """
        super(FMLayer, self).__init__()
        # 初始化参数
        self.concat_dim = concat_dim

        # 定义网络层
        self.fc = nn.Linear(concat_dim, 1)

        # 初始化网络层参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, stack, sparse):
        # 线性部分
        linear_part = self.fc(stack)

        # 特征交叉部分
        inter_part = torch.pow(torch.sum(sparse, dim=1), 2) - torch.sum(torch.pow(sparse, 2), dim=1)
        inter_part = 0.5 * torch.sum(inter_part, dim=-1, keepdim=True)

        return linear_part + inter_part


class DNN(nn.Module):
    """
    DNN层：多个全连接神经网络层，即多层感知机模型MLP
    """
    def __init__(self, hidden_units, dropout, output_dim):
        """
        :param hidden_units: Deep层隐藏层维度大小
        :param dropout: dropout层参数大小
        :param output_dim: Deep层输出维度大小
        """
        super(DNN, self).__init__()
        # 初始化参数
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.output_dim = output_dim

        # 定义网络层
        self.fc_layers = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_units[:-1], hidden_units[1:]):
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            self.fc_layers.append(nn.ReLU())
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_units[-1], output_dim)

        # 初始化网络层参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络参数"""
        for fc in self.fc_layers:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_normal_(fc.weight)
            if isinstance(fc, nn.Linear) and fc.bias is not None:
                fc.bias.data.zero_()

    def forward(self, x):
        for fc in self.fc_layers:
            x = fc(x)
        x = self.dropout_layer(x)
        output = self.output(x)
        return output


class DeepFM(nn.Module):
    """
    DeepFM模型：A Factorization-Machine based Neural Network
    """
    def __init__(self, embed_num, embed_dim, concat_dim, hidden_units, dropout, output_dim):
        """
        :param embed_num: 不同稀疏特征的嵌入数量
        :param embed_dim: embedding向量维度大小
        :param concat_dim: 连续特征和embedding后的稀疏特征拼接后的维度大小
        :param dropout: dropout层参数大小
        :param hidden_units: 深度网络层的隐藏层单元列表
        :param output_dim: Deep层输出维度大小
        """
        super(DeepFM, self).__init__()
        # 初始化参数
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.concat_dim = concat_dim
        self.dropout = dropout
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        # 定义网络层
        self.embed_layer = EmbedLayer(embed_num, embed_dim)
        self.fm_layer = FMLayer(concat_dim)
        self.hidden_layer = DNN(hidden_units, dropout, output_dim)

        # # 初始化网络层参数
        # self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        ...

    def forward(self, dense, sparse):
        # embedding layer
        batch = dense.shape[0]
        sparse_embed = self.embed_layer(sparse)
        stack = torch.cat((dense, sparse_embed.view(batch, -1)), dim=-1)

        # FM Layer
        fm_output = self.fm_layer(stack, sparse_embed)

        # Hidden Layer
        deep_output = self.hidden_layer(stack)

        # Output Units
        output = torch.sigmoid(fm_output + deep_output).squeeze()

        return output
