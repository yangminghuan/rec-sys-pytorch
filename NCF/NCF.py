"""
@Author: YMH
@Date: 2021-2-27
@Description: Neural network-based Collaborative Filtering论文模型代码复现
"""

import torch
import torch.nn as nn


class GMF(nn.Module):
    """
    广义矩阵分解模型
    """
    def __init__(self, user_num, item_num, embed_dim):
        """
        :param user_num: 用户数量
        :param item_num: 项目数量
        :param embed_dim: 嵌入层维度
        """
        super(GMF, self).__init__()
        # 初始化参数
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim

        # 定义网络层
        self.user_embed = nn.Embedding(user_num, embed_dim)
        self.item_embed = nn.Embedding(item_num, embed_dim)

        # 初始化网络层
        self._init_weight()

    def _init_weight(self):
        """初始化网络层"""
        nn.init.normal_(self.user_embed.weight, 0, 0.01)
        nn.init.normal_(self.item_embed.weight, 0, 0.01)

    def forward(self, user_id, item_id):
        user_vec = self.user_embed(user_id)
        item_vec = self.item_embed(item_id)
        # 进行element-wise product操作，充分交叉特征
        element_product = user_vec * item_vec
        return element_product


class MLP(nn.Module):
    """
    多层感知器模型
    """
    def __init__(self, user_num, item_num, embed_dim, layers):
        """
        :param user_num: 用户数量
        :param item_num: 项目数量
        :param embed_dim: 嵌入层维度
        :param layers: 线性层维度
        """
        super(MLP, self).__init__()
        # 初始化参数
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim

        # 定义网络层
        self.user_embed = nn.Embedding(user_num, embed_dim)
        self.item_embed = nn.Embedding(item_num, embed_dim)

        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        # 初始化网络层
        self._init_weight()

    def _init_weight(self):
        """初始化网络层"""
        nn.init.normal_(self.user_embed.weight, 0, 0.01)
        nn.init.normal_(self.item_embed.weight, 0, 0.01)

        for fc in self.fc_layers:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_normal_(fc.weight)
            if isinstance(fc, nn.Linear) and fc.bias is not None:
                fc.bias.data.zero_()

    def forward(self, user_id, item_id):
        user_vec = self.user_embed(user_id)
        item_vec = self.item_embed(item_id)
        mlp_vec = torch.cat([user_vec, item_vec], dim=-1)

        for fc in self.fc_layers:
            mlp_vec = fc(mlp_vec)
        return mlp_vec


class NCF(nn.Module):
    """
    结合神经网络的协同过滤模型
    """
    def __init__(self, user_num, item_num, embed_dim, layers):
        """
        :param user_num: 用户数量
        :param item_num: 项目数量
        :param embed_dim: 嵌入层维度
        :param layers: 线性层维度
        """
        super(NCF, self).__init__()
        # 初始化参数
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.layers = layers

        # 定义网络层
        self.GMF_layer = GMF(user_num, item_num, embed_dim)
        self.MLP_layer = MLP(user_num, item_num, embed_dim, layers)
        self.output = nn.Linear(embed_dim + layers[-1], 1)
        self.logistic = nn.Sigmoid()

        # 初始化网络层
        self._init_weight()

    def _init_weight(self):
        """初始化网络层"""
        nn.init.xavier_normal_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, user_id, item_id):
        gmf_vec = self.GMF_layer(user_id, item_id)
        mlp_vec = self.MLP_layer(user_id, item_id)

        ncf_vec = torch.cat([gmf_vec, mlp_vec], dim=-1)
        output = self.output(ncf_vec)
        output = self.logistic(output)

        return output.squeeze()
