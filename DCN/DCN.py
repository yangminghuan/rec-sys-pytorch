"""
@Author: YMH
@Date:
@Description: Deep & Cross Network for Ad Click Predictions论文模型代码复现
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


class CrossNetwork(nn.Module):
    """
    Cross Network层：进行特征之间的交叉组合，构造有限高阶交叉特征，增加特征之间的交互力度
    """
    def __init__(self, input_dim, layer_num):
        """
        :param input_dim: 连续特征和稀疏特征embedding向量拼接后的输入维度
        :param layer_num: 交叉层数
        """
        super(CrossNetwork, self).__init__()
        # 初始化参数
        self.input_dim = input_dim
        self.layer_num = layer_num

        # 定义网络层
        self.cross_w = nn.Parameter(torch.randn(layer_num, input_dim))
        self.cross_bias = nn.Parameter(torch.randn(layer_num, input_dim))
        self.BatchNorm_list = nn.ModuleList()
        for _ in range(layer_num):
            self.BatchNorm_list.append(nn.BatchNorm1d(input_dim))

        # 初始化网络层参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        nn.init.xavier_normal_(self.cross_w)
        nn.init.zeros_(self.cross_bias)

    def forward(self, x):
        x_cross = x
        for i in range(self.layer_num):
            w = torch.unsqueeze(self.cross_w[i, :].T, dim=1)
            xt_w = torch.mm(x, w)
            x_cross = x_cross * xt_w + self.cross_bias[i, :] + x_cross
            x_cross = self.BatchNorm_list[i](x_cross)
        return x_cross


class DeepNetwork(nn.Module):
    """
    Deep Network层：一个普通的全连接神经网络层，即多层感知机模型MLP
    """
    def __init__(self, hidden_units, output_dim):
        """
        :param hidden_units: Deep层隐藏层维度大小
        :param output_dim: Deep层输出维度
        """
        super(DeepNetwork, self).__init__()
        # 初始化参数
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        # 定义网络层
        self.fc_layers = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_units[:-1], hidden_units[1:]):
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            self.fc_layers.append(nn.ReLU())
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
        output = self.output(x)
        return output


class DCN(nn.Module):
    """
    Deep & Cross Network模型：深度交叉网络模型（DCN）
    """
    def __init__(self, embed_num, embed_dim, input_dim, layer_num, hidden_units, output_dim):
        """
        :param embed_num: 不同稀疏特征的嵌入数量
        :param embed_dim: embedding向量维度大小
        :param input_dim: 连续特征和稀疏特征嵌入向量拼接后的输入维度
        :param layer_num: 交叉网络层数
        :param hidden_units: 深度网络层的隐藏层单元列表
        :param output_dim: 深度网络层的输出维度
        """
        super(DCN, self).__init__()
        # 初始化参数
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.layer_num = layer_num
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        # 定义网络层
        self.embed_layer = EmbedLayer(embed_num, embed_dim)
        self.cross_layer = CrossNetwork(input_dim, layer_num)
        self.deep_layer = DeepNetwork(hidden_units, output_dim)
        self.output_layer = nn.Linear(input_dim + output_dim, 1)

        # 初始化网络层参数
        self._init_weight()

    def _init_weight(self):
        """初始化网络层参数"""
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, dense, sparse):
        # embedding and stacking layer
        sparse_embed = self.embed_layer(sparse)
        input_data = torch.cat((dense, sparse_embed), dim=-1)

        # Cross network
        x_cross = self.cross_layer(input_data)

        # Deep network
        x_deep = self.deep_layer(input_data)

        # Combination output layer
        x_stack = torch.cat((x_cross, x_deep), dim=-1)
        output = torch.sigmoid(self.output_layer(x_stack)).squeeze()

        return output
