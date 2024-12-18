import os
import pickle
import random
import pandas as pd

from gensim.models import Word2Vec
import os.path as osp

class DeepWalkEmbedding():
    def __init__(self, 
                 graph_path,
                 index_userid_path,
                 index_songid_path, 
                 num_walk, 
                 sequence_len, 
                 embedding_size=16, 
                 window_size=5):
        """_summary_

        Args:
            graph_path (_type_): 保存graph的pkl的路径
            num_walk (_type_): 每个节点生成几条路径
            sequence_len (_type_): 每个节点生成多长的路径
            embedding_size: 每个节点embedding的长度
        """
        self.graph = pickle.load(open(graph_path, 'rb'))
        self.all_nodes = list(self.graph.keys())
        self.num_walk = num_walk
        self.sequence_len = sequence_len
        self.embedding_size = embedding_size
        self.window_size = window_size

        self.index_userid_dict = pickle.load(open(index_userid_path, 'rb'))
        self.index_songid_dict = pickle.load(open(index_songid_path, 'rb'))

        self.user_node_num = len(self.index_userid_dict.keys())
        self.song_node_num = len(self.index_songid_dict.keys())

        self.all_sequence = []

    def _generate_sequence_by_random_walk(self):
        """
        生成随机游走序列
        """
        for i in range(self.num_walk):
            random.shuffle(self.all_nodes)
            for node in self.all_nodes:
                # word2vec模型需要使用字符串序列
                sequence = [str(node)]
                while len(sequence) < self.sequence_len:
                    neighboors = self.graph[node]
                    if len(neighboors) == 0:
                        break
                    else:
                        # 随机选择一个邻居节点
                        nei = random.choice(neighboors)
                        # 更新node的值
                        node = nei
                        sequence.append(str(nei))
                self.all_sequence.append(sequence)

    def train_word2vec_model(self):
        print('=====> Training word2vec model...')

        self._generate_sequence_by_random_walk()
        self.word2vec_model = Word2Vec(sentences=self.all_sequence, 
                         vector_size=self.embedding_size,
                         window=self.window_size,
                         min_count=1,
                         sg=1,   # 表示使用CBOW（Continuous Bag of Words）算法还是Skip-gram算法，0表示CBOW，1表示Skip-gram
                         workers=4)
        
        print('=====> Finish training word2vec model!')
    
    # 保存所有的user embedding
    def save_users_embedding(self, path):
        userid_embedding_dict = {}
        for i in range(self.user_node_num):
            userid_embedding_dict[self.index_userid_dict[i]] = self.word2vec_model.wv[str(i)]
        pickle.dump(userid_embedding_dict, open(path, 'wb'))

    # 保存所有的song embedding
    def save_songs_embedding(self, path):
        songid_embedding_dict = {}
        for i in range(self.user_node_num, self.user_node_num + self.song_node_num):
            songid_embedding_dict[self.index_songid_dict[i]] = self.word2vec_model.wv[str(i)]
        pickle.dump(songid_embedding_dict, open(path, 'wb'))
    



if __name__ == '__main__':
    basedir = '/data/zhy/recommendation_system/musicRec'

    graph_path = osp.join(basedir, 'cache/graph.pkl')
    index_userid_path = osp.join(basedir, 'cache/index_userid_dict.pkl')
    index_songid_path = osp.join(basedir, 'cache/index_songid_dict.pkl')

    deep_walk_embedding = DeepWalkEmbedding(graph_path, 
                                            index_userid_path, 
                                            index_songid_path, 
                                            num_walk=5, 
                                            sequence_len=10,
                                            embedding_size=16, 
                                            window_size=5)
    
    deep_walk_embedding.train_word2vec_model()

    deep_walk_embedding.save_users_embedding(osp.join(basedir, 'cache/users_embedding.pkl'))
    deep_walk_embedding.save_songs_embedding(osp.join(basedir, 'cache/songs_embedding.pkl'))