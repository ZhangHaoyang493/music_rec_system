import os
import pickle
import os.path as osp
import pandas as pd

class HotSongsRecall():
    def __init__(self, train_data_path, basedir='/Users/zhanghaoyang04/codeSpace/music_rec_system'):
        train_data = pd.read_csv(train_data_path)

        # 根据song_id聚类，统计每首歌曲被听的次数
        song_counts = train_data['song_id'].value_counts().reset_index()
        song_counts.columns = ['song_id', 'listen_count']

        self.songs_count_dict = dict(zip(song_counts['song_id'], song_counts['listen_count']))
        pickle.dump(self.songs_count_dict, open(osp.join(basedir, 'cache/hot_songs.pkl'), 'wb'))

    def recall(self, k=10):
        """
        k: 召回几个热门音乐物料
        """
        return list(self.songs_count_dict.keys())[:k]

if __name__ == '__main__':
    recall_model = HotSongsRecall('/Users/zhanghaoyang04/codeSpace/music_rec_system/data_for_test/train.csv')

    print(recall_model.recall(10))
