"""
@Author: YMH
@Date:
@Description: BPR: Bayesian Personalized Ranking from Implicit Feedback模型代码复现
"""

import torch.nn as nn


class BPR(nn.Module):
    """
    贝叶斯个性化排序模型（基于矩阵分解模型）
    """
    def __init__(self, user_num, item_num, embed_dim):
        """
        :param user_num: 用户数
        :param item_num: 项目数
        :param embed_dim: 嵌入层维度
        :return:
        """
        super(BPR, self).__init__()
        # 初始化参数
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim

        # 定义模型层
        self.user_embed = nn.Embedding(user_num, embed_dim)
        self.item_embed = nn.Embedding(item_num, embed_dim)

        # 初始化模型层
        self._init_weight()

    def _init_weight(self):
        """
        初始化模型层
        """
        nn.init.normal_(self.user_embed.weight, 0, 0.01)
        nn.init.normal_(self.item_embed.weight, 0, 0.01)

    def forward(self, user_id, pos_id, neg_id):
        """
        :param user_id: 用户id
        :param pos_id: 正样本id
        :param neg_id: 负样本id
        :return: 返回用户更加偏好正样本的程度
        """
        user_vec = self.user_embed(user_id)
        pos_vec = self.item_embed(pos_id)
        neg_vec = self.item_embed(neg_id)

        pos_score = (user_vec * pos_vec).sum(dim=-1)
        neg_score = (user_vec * neg_vec).sum(dim=-1)

        return pos_score, neg_score
