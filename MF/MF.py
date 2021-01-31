"""
@Author: YMH
@Date: 2021-1-31
@Description: Matrix Factorization Techniques for Recommender Systems模型代码复现
"""

import torch.nn as nn


class MF(nn.Module):
    """
    矩阵分解推荐模型
    """
    def __init__(self, user_num, item_num, hidden_dim):
        """
        :param user_num: 用户数量
        :param item_num: 物品数量
        :param hidden_dim: 隐向量维度
        """
        super(MF, self).__init__()
        # 初始化参数
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_dim = hidden_dim

        # 定义模型层
        self.user_embed = nn.Embedding(user_num, hidden_dim)
        self.item_embed = nn.Embedding(item_num, hidden_dim)

        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)

        # 初始化模型层
        self._init_weight()

    def _init_weight(self):
        """
        初始化模型层
        """
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)
        nn.init.normal_(self.user_bias.weight, 0, 0.01)
        nn.init.normal_(self.item_bias.weight, 0, 0.01)

    def forward(self, user_id, item_id, avg_score):
        """
        :param user_id: 用户id
        :param item_id: 物品id
        :param avg_score: 用户的平均评分
        :return: 用户对物品的评分
        """
        user_vec = self.user_embed(user_id)
        item_vec = self.item_embed(item_id)
        user_bias = self.user_bias(user_id).view(-1)
        item_bias = self.item_bias(item_id).view(-1)

        rating = (user_vec * item_vec).sum(dim=1)
        rating = rating + avg_score + user_bias + item_bias

        return rating
