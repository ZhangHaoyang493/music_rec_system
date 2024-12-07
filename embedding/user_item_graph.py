"""
生成用户和物品点击关系的图

图的表示方法：
邻接表（使用字典保存，key为节点，value为邻居节点列表）
"""

import pandas as pd
import pickle

train_data = pd.read_csv('../data_for_test_scripts/train.csv')
members = pd.read_csv('../data_for_test_scripts/members.csv')
songs = pd.read_csv('../data_for_test_scripts/songs.csv')

# 获取所有用户的字符串id和所有歌曲的字符串id
all_user_id = members['msno'].unique().tolist()
all_song_id = songs['song_id'].unique().tolist()

# 读取用户映射为id的字典和歌曲映射为id的字典
user_to_num_dict = pickle.load(open('../cache/user_id_dict.pkl', 'rb'))
song_to_num_dict = pickle.load(open('../cache/song_id_dict.pkl', 'rb'))

graph = {}

for userid in all_user_id:
    graph[(0, user_to_num_dict[userid])] = []  # 用户的id用0标识
for song_id in all_song_id:
    graph[(1, song_to_num_dict[song_id])] = []  # 歌曲的id用1标识

# 从用户听歌日志中将用户和歌曲的点击关系加入图中
user_song_list = train_data[['msno', 'song_id']].values.tolist()
for user_song in user_song_list:
    graph[(0, user_to_num_dict[user_song[0]])].append((1, song_to_num_dict[user_song[1]]))
    graph[(1, song_to_num_dict[user_song[1]])].append((0, user_to_num_dict[user_song[0]]))

with open('../cache/graph.pkl', 'wb') as f:
    pickle.dump(graph, f)