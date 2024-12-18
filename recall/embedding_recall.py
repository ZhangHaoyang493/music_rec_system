import os
import pickle
import faiss
import numpy as np


class EmbeddingRecall():
    def __init__(self, user_embedding_path, song_embedding_path):
        self.user_embedding_dict = pickle.load(open(user_embedding_path, 'br'))
        self.song_embedding_dict = pickle.load(open(song_embedding_path, 'br'))

        self.songEmbedding_index_dict = dict(
            zip(self.song_embedding_dict.keys(), range(len(self.song_embedding_dict.keys())))
        )
        self.index_songEmbedding_dict = dict(
            zip(range(len(self.song_embedding_dict.keys())), self.song_embedding_dict.keys())
        )

        # 将song_embedding存入faiss数据库
        song_embedding = np.array(list(self.song_embedding_dict.values()))
        self.song_embedding_faiss = faiss.IndexFlatL2(song_embedding.shape[-1])
        self.song_embedding_faiss.add(song_embedding)

        print('Size of faiss index', self.song_embedding_faiss.ntotal)

    def embedding_recall_for_userid(self, user_id, k=5):
        """_summary_

        Args:
            user_id (_type_): 用户
            k (int, optional): _description_. Defaults to 5.
        """
        query = np.array([self.user_embedding_dict[user_id]])
        _, indices = self.song_embedding_faiss.search(query, k)
        # 将获取的index转为真正的songid
        res_song_id = [self.index_songEmbedding_dict[i] for i in indices[0]]

        return res_song_id


if __name__ == '__main__':
    embedding_recall = EmbeddingRecall(
        '/data/zhy/recommendation_system/musicRec/cache/users_embedding.pkl',
        '/data/zhy/recommendation_system/musicRec/cache/songs_embedding.pkl'
    )
    
    recall_songs_id = embedding_recall.embedding_recall_for_userid('XQxgAYj3klVKjR3oxPPXYYFp4soD4TuBghkhMTD4oTw=')
    print(recall_songs_id)





