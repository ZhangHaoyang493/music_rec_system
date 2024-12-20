import os
import pickle
import pandas as pd

"""
利用倒排索引进行策略召回，召回包括演唱者、作曲者、作词者、歌曲类型以及歌曲语言的召回
再加一个召回方式：用户在一个月内反复听过的歌曲选择召回
"""

class StrategyRecall():
    def __init__(self, bs_path, train_data_path, song_data_path):
        self.bs = pickle.load(open(bs_path, 'rb'))
        self.train_data = pd.read_csv(train_data_path)
        self.song_data = pd.read_csv(song_data_path)

        self.all_data = pd.merge(self.train_data, self.song_data, on='song_id', how='left')

        # 获取历史听过的歌手名字
        self.user_history_songer_list_dict = self.all_data.groupby('msno')['artist_name'].apply(set).to_dict()
        self.user_history_composer_list_dict = self.all_data.groupby('msno')['composer'].apply(set).to_dict()
        self.user_history_lyricist_list_dict = self.all_data.groupby('msno')['lyricist'].apply(set).to_dict()
        self.user_history_language_list_dict = self.all_data.groupby('msno')['language'].apply(set).to_dict()
        self.user_history_cate_list_dict = self.all_data.groupby('msno')['genre_ids'].apply(set).to_dict()
        
        # 获取听歌的历史列表
        def _history_song_select(x):
            # 如果target=1，说明这个歌曲被用户反复听过，值得再次去推荐
            # 否则，不值得再次去推荐

            # 保存结果，key为songid，value为target
            history_dict = {}
            for songid, targrt in zip(x['song_id'], x['target']):
                history_dict[songid] = targrt

            return history_dict

        self.user_history_songs = self.all_data.groupby('msno').apply(_history_song_select).to_dict()

        # print(self.user_history_songs['XQxgAYj3klVKjR3oxPPXYYFp4soD4TuBghkhMTD4oTw='])

    def songer_recall(self, userid):
        songer_history = self.user_history_songer_list_dict[userid]

        recall_list = []
        for songer in songer_history:
            recall_list += self.bs['songer'][songer]
            
        return list(set(recall_list))

    def composer_recall(self, userid):
        composer_history = self.user_history_composer_list_dict[userid]

        recall_list = []
        for composer in composer_history:
            recall_list += self.bs['composer'][composer]

        return list(set(recall_list))

    def lyricist_recall(self, userid):
        lyricist_history = self.user_history_lyricist_list_dict[userid]

        recall_list = []
        for lyricist in lyricist_history:
            recall_list += self.bs['lyricist'][lyricist]
        
        return list(set(recall_list))
    
    def language_recall(self, userid):
        language_history = self.user_history_language_list_dict[userid]

        recall_list = []
        for language in language_history:
            recall_list += self.bs['language'][language]
        
        return list(set(recall_list))

    def cate_recall(self, userid):
        cate_history = self.user_history_cate_list_dict[userid]

        recall_list = []
        for cate in cate_history:
            recall_list += self.bs['category'][cate]

        return list(set(recall_list))
    
    def history_recall(self, userid):
        recall_list = []

        for his, target in self.user_history_songs[userid].items():
            # 如果target为1，说明用户喜欢反复听，可以召回
            if target == 1:
                recall_list.append(his)

        return list(set(recall_list))
    
    def strategy_recall(self, userid):
        recall_list = []

        recall_list += self.songer_recall(userid)
        recall_list += self.composer_recall(userid)
        recall_list += self.lyricist_recall(userid)
        recall_list += self.language_recall(userid)
        recall_list += self.cate_recall(userid)
        recall_list += self.history_recall(userid)

        recall_list = list(set(recall_list))

        final_recall_list = []
        for song in recall_list:
            if song in self.user_history_songs[userid]:
                if self.user_history_songs[userid][song] == 0:
                    continue    
            final_recall_list.append(song)

        return final_recall_list


if __name__ == '__main__':
    songer_recall = StrategyRecall(
        '/data/zhy/recommendation_system/musicRec/cache/bs.pkl',
        '/data/zhy/recommendation_system/musicRec/data_for_test/train.csv',
        '/data/zhy/recommendation_system/musicRec/data_for_test/songs.csv'
    )

    print(songer_recall.strategy_recall('N+dxmo1qvkKAlzYtcxGzjrqcpVyX9J7AMlXFuYASKuY='))


    
