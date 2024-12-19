# 构建倒排索引
"""
构建一个字典，key为songer、composer、lyrics、category，language，
分别存储歌手、作曲者、作词者、类型以及语言的倒排索引
"""
import pandas as pd
import os.path as osp
import pickle

basedir = '/data/zhy/recommendation_system/musicRec'

bs = {}


songs_data = pd.read_csv(osp.join(basedir, 'data_for_test/songs.csv'))

bs['songer'] = songs_data.groupby('artist_name')['song_id'].apply(list).to_dict()
bs['composer'] = songs_data.groupby('composer')['song_id'].apply(list).to_dict()
bs['lyricist'] = songs_data.groupby('lyricist')['song_id'].apply(list).to_dict()
bs['category'] = songs_data.groupby('genre_ids')['song_id'].apply(list).to_dict()
bs['language'] = songs_data.groupby('language')['song_id'].apply(list).to_dict()

pickle.dump(bs, open(osp.join(basedir, 'cache/bs.pkl'), 'wb'))