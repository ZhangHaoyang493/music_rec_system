import os
import pickle

import torch
import torch.nn as nn
import lightning as L

from torch.utils.data import Dataset, DataLoader

class Gaph2Vec(L.LightningModule):
    def __init__(self, user_num, item_num, dim=32, window_size=5, lr=0.01):
        """_summary_

        Args:
            user_num (_type_): 用户数量
            item_num (_type_): 物料数量
            dim (int, optional): embedding的维度. Defaults to 32.
            window_size (int, optional): 滑窗大小. Defaults to 5.
            lr: 学习率
        """
        super().__init__()

        self.user_embedding = nn.Embedding(user_num, dim)
        self.item_embedding = nn.Embedding(item_num, dim)

        self.user_num = user_num
        self.item_num = item_num
        self.dim = dim
        self.window_size = window_size
        self.lr = lr

    # 获取x对应的embedding
    def _get_embedding(self, x):
        # 如果x[0]是0，说明这是一个用户的id，从用户的embedding中获取embedding
        if x[0] == 0:
            return self.user_embedding(x[1])
        else:
            return self.item_embedding(x[1])

    def training_step(self, batch, batch_idx):
        pos_data = batch['pos']   # [(0/1, id), (0/1, id), (0/1, id), (0/1, id), (0/1, id)]
        neg_data = batch['neg']   # [(0/1, neg_id), (0/1, neg_id), (0/1, neg_id), (0/1, neg_id),3 (0/1, neg_id)]

        # 获取所有正样本的embedding，负样本的embedding以及目标物品的embedding
        target_embedding = None
        pos_embedding, neg_embedding = [], []
        for i, pos in enumerate(pos_data):
            if i == self.window_size // 2:
                target_embedding = self._get_embedding(pos)
                continue
            pos_embedding.append(self._get_embedding(pos))
        for neg in neg_data:
            neg_embedding.append(self._get_embedding(neg))

        return self.NEG_loss(target_embedding, pos_embedding, neg_embedding)

    def NEG_loss(self, target_embedding, pos_embedding, neg_embedding):
        # 获取正样本对的个数
        pos_num = len(pos_embedding)
        
        loss = 0.0
        # NEG loss最简单的实现方式，未优化
        # 对于每一个正样本
        for pos_emb in pos_embedding:
            loss += torch.log(1 + torch.exp(-torch.dot(pos_emb, target_embedding)))
            # 对于每一个负样本
            for neg_emb in neg_embedding:
                loss += torch.log(1 + torch.exp(torch.dot(neg_emb, target_embedding)))
        loss /= pos_num

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def get_user_embedding(self):
        user_embedding = self.user_embedding.weight.data.detach().cpu().numpy().tolist()
        user_embedding_dict = dict(zip(range(self.user_num), user_embedding))

        # key: 用户id
        # value：用户的embedding
        return user_embedding_dict

    def get_item_embedding(self):
        item_embedding = self.item_embedding.weight.data.detach().cpu().numpy().tolist()
        item_embedding_dict = dict(zip(range(self.item_num), item_embedding))
        
        return item_embedding_dict
    

class Graph2vecDataset(Dataset):
    def __init__(self):
        super().__init__()

        assert os.path.exists('../cache/graph.pkl'), '请先运行user_item_graph.py得到图文件'
        self.graph = pickle.load(open('../cache/graph.pkl', 'rb'))


    
    
