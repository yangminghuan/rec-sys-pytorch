"""
@Author: YMH
@Date: 2021-4-9
@Description: Neural Factorization Machines for Sparse Predictive Analytics论文模型代码复现
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


class DNN(nn.Module):
    """
    DNN层：多个全连接神经网络层，即多层感知机模型MLP
    """
    def __init__(self, hidden_units, dropout):
        """
        :param hidden_units: Deep层隐藏层维度大小
        :param dropout: dropout层参数大小
        """
        super(DNN, self).__init__()
        # 初始化参数
        self.hidden_units = hidden_units
        self.dropout = dropout

        # 定义网络层
        self.fc_layers = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_units[:-1], hidden_units[1:]):
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            self.fc_layers.append(nn.ReLU())
        self.dropout_layer = nn.Dropout(dropout)

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
        output = self.dropout_layer(x)
        return output


class NFM(nn.Module):
    """
    Neural Factorization Machines模型：神经因子分解机模型（NFM）
    """
    def __init__(self, embed_num, embed_dim, dense_dim, hidden_units, dropout):
        """
        :param embed_num: 不同稀疏特征的嵌入数量
        :param embed_dim: embedding向量维度大小
        :param dense_dim: 连续特征维度大小
        :param dropout: dropout层参数大小
        :param hidden_units: 深度网络层的隐藏层单元列表
        """
        super(NFM, self).__init__()
        # 初始化参数
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.hidden_units = hidden_units

        # 定义网络层
        self.embed_layer = EmbedLayer(embed_num, embed_dim)
        self.bn = nn.BatchNorm1d(dense_dim + embed_dim)
        self.dnn_layer = DNN(hidden_units, dropout)
        self.output_layer = nn.Linear(hidden_units[-1], 1)

        # 初始化网络层参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, dense, sparse):
        # embedding layer
        sparse_embed = self.embed_layer(sparse)

        # Bi-Interaction layer
        inter_embed = 0.5 * (torch.pow(torch.sum(sparse_embed, dim=1), 2) -
                             torch.sum(torch.pow(sparse_embed, 2), dim=1))

        # concat
        x = torch.cat((dense, inter_embed), dim=-1)
        x = self.bn(x)

        # Hidden layer
        x = self.dnn_layer(x)

        # Prediction score
        output = torch.sigmoid(self.output_layer(x)).squeeze()

        return output
