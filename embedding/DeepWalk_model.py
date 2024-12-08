import os
import pickle
import random
import pandas as pd

import torch
import torch.nn as nn
import lightning as L
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import Dataset, DataLoader

base_dir = '/data/zhy/recommendation_system/musicRec'


class DeepWalk(L.LightningModule):
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
    

class DeepWalkDataset(Dataset):
    def __init__(self, window_size=5, sequence_len=10, num_walk=5, neg_sample_num=5):
        """_summary_

        Args:
            window_size (int, optional): word2vec的窗口大小. Defaults to 5.
            sequence_len (int, optional): 每个序列的长度. Defaults to 10.
            num_walk (int, optional): 对于每个节点，生成多少个序列. Defaults to 5.
            neg_sample_num: 对于每个正样本，采样多少个负样本
        """
        super().__init__()

        assert os.path.exists('%s/cache/graph.pkl' % base_dir), '请先运行user_item_graph.py得到图文件'
        self.graph = pickle.load(open('%s/cache/graph.pkl' % base_dir, 'rb'))

        assert sequence_len >= window_size

        self.all_sequence = []
        self.window_size = window_size
        self.sequence_len = sequence_len
        self.num_walk = num_walk
        self.neg_sample_num = neg_sample_num

        # 获取所有图节点
        user_nodes: dict = pickle.load(open('%s/cache/user_id_dict.pkl' % base_dir, 'rb'))
        self.user_nodes = list(user_nodes.values())
        song_nodes: dict = pickle.load(open('%s/cache/song_id_dict.pkl' % base_dir, 'rb'))
        self.song_nodes = list(song_nodes.values())
        self.all_nodes = [(0, i) for i in self.user_nodes] + [(1, i) for i in self.song_nodes]
        

        # random walk生成序列
        self._generate_sequence_by_random_walk()
        self._generate_all_datas()

    def _generate_sequence_by_random_walk(self):
        """
        生成随机游走序列
        """
        self.all_sequence = []
        
        for i in range(self.num_walk):
            random.shuffle(self.all_nodes)
            for node in self.all_nodes:
                sequence = [(node[0], torch.tensor([node[1]]))]
                while len(sequence) < self.sequence_len:
                    neighboors = self.graph[node]
                    if len(neighboors) == 0:
                        break
                    else:
                        nei = random.choice(neighboors)
                        sequence.append((nei[0], torch.tensor([nei[1]])))
                self.all_sequence.append(sequence)
    
    def _generate_all_datas(self):
        """
        根据所有的sequence生成和window_size一样长的所有数据，然后进行训练
        """
        self.samples = []

        for seq in self.all_sequence:
            if len(seq) < self.window_size:
                assert len(seq) == 1
                self.samples.append([seq[0] for i in range(self.window_size)])
            else:
                self.samples.extend([seq[i: i+self.window_size] 
                                     for i in range(self.sequence_len - self.window_size)])
        

    def __getitem__(self, index):
        pos = self.samples[index]
        neg = random.choices(self.all_nodes, k=self.neg_sample_num)

        return {'pos': pos, 'neg': neg}
    

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    user_num = len(pickle.load(
        open('%s/cache/user_id_dict.pkl' % base_dir, 'rb')).keys())
    song_num = len(pickle.load(
        open('%s/cache/song_id_dict.pkl' % base_dir, 'rb')).keys())

    model = DeepWalk(user_num, song_num, 16, 5, 0.01)

    dataset = DeepWalkDataset(window_size=5, sequence_len=10, num_walk=5, neg_sample_num=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    logger = TensorBoardLogger('deep_walk', name='DeepWalk')
    trainer = L.Trainer(max_epochs=10, accelerator='gpu', devices=1, logger=logger)

    trainer.fit(model, dataloader)
