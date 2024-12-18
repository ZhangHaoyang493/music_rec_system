import os
import pickle
import pandas as pd

class SongerRecall():
    def __init__(self, user_feature_path, songs_path):
        self.user_feature_path = pd.read_csv(user_feature_path)
        self.songs = pd.read_csv(songs_path)

        self.user_songer_dict = dict(
            zip(
                self.user_feature_path['msno'].to_list(), 
                self.user_feature_path['favourite_top_5_artist_name'].to_list()
            )
        )

        self.songer_songid_dict = dict(
            zip(
                self.songs['song_id'].to_list(),
                self.songs['artist_name'].to_list()
            )
        )

        print(self.user_songer_dict)




if __name__ == '__main__':
    songer_recall = SongerRecall('/data/zhy/recommendation_system/musicRec/cache/user_feature_for_recall.csv')


    
