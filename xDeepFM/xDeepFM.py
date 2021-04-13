"""
@Author: YMH
@Date: 2021-4-13
@Description: Combining Explicit and Implicit Feature Interactions for Recommender Systems论文模型代码复现
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


class CIN(nn.Module):
    """
    Compressed Interaction Network(CIN)压缩交互网络采用vector-wise方式对特征向量进行交叉组合，类比DCN中的cross network
    """
    def __init__(self, num_field, cin_layers, split_half, output_dim):
        """
        :param num_field: 稀疏特征域数量
        :param cin_layers: CIN隐藏层列表
        :param split_half: 是否将输出结果与保留的隐藏层结果分割开来（默认为True）
        :param output_dim: CIN输出层维度
        """
        super(CIN, self).__init__()
        # 初始化参数
        self.num_field = num_field
        self.cin_layers = cin_layers
        self.split_half = split_half
        self.cin_layer_dims = [num_field] + cin_layers

        # 定义网络层
        prev_dim = num_field
        fc_dim = 0
        self.conv1d_layers = nn.ModuleList()
        for k in range(1, len(self.cin_layer_dims)):
            self.conv1d_layers.append(nn.Conv1d(self.cin_layer_dims[0] * prev_dim, self.cin_layer_dims[k], 1))
            if split_half and k != len(self.cin_layers):
                prev_dim = self.cin_layer_dims[k] // 2
            else:
                prev_dim = self.cin_layer_dims[k]
            fc_dim += prev_dim
        self.cin_output = nn.Linear(fc_dim, output_dim)

        # 初始化网络层参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        for conv1d in self.conv1d_layers:
            nn.init.xavier_uniform_(conv1d.weight)
        nn.init.xavier_normal_(self.cin_output.weight)
        self.cin_output.bias.data.zero_()

    def forward(self, sparse_embed):
        batch, embed_dim = sparse_embed.shape[0], sparse_embed.shape[2]
        # 初始化隐藏层列表
        x_list = [sparse_embed]
        result = []

        for k in range(1, len(self.cin_layer_dims)):
            # 上一层矩阵X_k和原始矩阵X_0进行向量外积计算
            z_k = torch.einsum('bhd,bmd->bhmd', x_list[-1], sparse_embed)
            z_k = z_k.reshape(batch, x_list[-1].shape[1] * self.num_field, embed_dim)

            # 利用不同参数矩阵对z_k进行加权求和，类似卷积操作
            x_k = self.conv1d_layers[k - 1](z_k)
            x_k = torch.relu(x_k)

            if self.split_half and k != len(self.cin_layers):
                next_x, x_k = torch.split(x_k, x_k.shape[1] // 2, 1)
            else:
                next_x, x_k = x_k, x_k

            x_list.append(next_x)
            result.append(x_k)

        result = torch.cat(result, dim=1)
        result = torch.sum(result, dim=2)
        result = self.cin_output(result)

        return result


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
        nn.init.xavier_normal_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, x):
        for fc in self.fc_layers:
            x = fc(x)
        x = self.dropout_layer(x)
        output = self.output(x)
        return output


class xDeepFM(nn.Module):
    """
    xDeepFM模型：Combining Explicit and Implicit Feature Interactions
    """
    def __init__(self, embed_num, embed_dim, cin_layers, hidden_units, dropout, output_dim, split_half=True):
        """
        :param embed_num: 稀疏特征各域类别数量
        :param embed_dim: 稀疏特征嵌入向量维度
        :param cin_layers: CIN隐藏层维度列表
        :param hidden_units: Deep隐藏层维度列表
        :param dropout: dropout层维度
        :param output_dim: 各稀疏特征交叉层输出维度大小
        :param split_half: 是否将CIN层输出结果与保留的隐藏层结果分割开来（默认为True）
        """
        super(xDeepFM, self).__init__()
        # 初始化参数
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.cin_layers = cin_layers
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.output_dim = output_dim
        self.split_half = split_half

        # 定义网络层
        self.embed_layer = EmbedLayer(embed_num, embed_dim)
        self.linear = nn.Linear(hidden_units[0], 1)
        num_field = len(embed_num)
        self.CIN = CIN(num_field, cin_layers, split_half, output_dim)
        self.Plain_DNN = DNN(hidden_units, dropout, output_dim)

        # 初始化网络层参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        nn.init.xavier_normal_(self.linear.weight)
        self.linear.bias.data.zero_()

    def forward(self, dense, sparse):
        # Embedding layer
        batch = dense.shape[0]
        sparse_embed = self.embed_layer(sparse)
        stack = torch.cat((dense, sparse_embed.reshape(batch, -1)), dim=-1)

        # Linear
        linear_output = self.linear(stack)

        # CIN
        cin_output = self.CIN(sparse_embed)

        # Plain DNN
        dnn_output = self.Plain_DNN(stack)

        # Output unit
        output = torch.sigmoid(linear_output + cin_output + dnn_output).squeeze()

        return output
