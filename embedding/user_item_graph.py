"""
先生成用户以及歌曲和index的一一对应的字典并保存

再生成用户和歌曲点击关系的图并保存

图的表示方法：
邻接表（使用字典保存，key为节点，value为邻居节点列表）
"""

import pandas as pd
import pickle
import os


basedir = '/data/zhy/recommendation_system/musicRec'

train_data = pd.read_csv(os.path.join(basedir, 'data_for_test/train.csv'))
members = pd.read_csv(os.path.join(basedir, 'data_for_test/members.csv'))
songs = pd.read_csv(os.path.join(basedir, 'data_for_test/songs.csv'))

# 获取所有用户的字符串id和所有歌曲的字符串id
all_user_id = members['msno'].unique().tolist()
all_song_id = songs['song_id'].unique().tolist()

# 生成userid、songid和index的对应字典
# 0 ~ len(all_user_id) 的index编号给userid
# len(all_user_id) ~ len(all_user_id) + len(all_song_id) 的index编号给songid
userid_index_dict = dict(zip(all_user_id, range(len(all_user_id))))
songid_index_dict = dict(zip(all_song_id, range(len(all_user_id), len(all_user_id) + len(all_song_id))))

index_userid_dict = {}
index_songid_dict = {}
# 生成index和userid以及songid的对应字典
for k, v in userid_index_dict.items():
    index_userid_dict[v] = k
for k, v in songid_index_dict.items():
    index_songid_dict[v] = k

# 保存字典
pickle.dump(userid_index_dict, open(os.path.join(basedir, 'cache/userid_index_dict.pkl'), 'bw'))
pickle.dump(songid_index_dict, open(os.path.join(basedir, 'cache/songid_index_dict.pkl'), 'bw'))
pickle.dump(index_userid_dict, open(os.path.join(basedir, 'cache/index_userid_dict.pkl'), 'bw'))
pickle.dump(index_songid_dict, open(os.path.join(basedir, 'cache/index_songid_dict.pkl'), 'bw'))


graph = {}

for userid in all_user_id:
    graph[userid_index_dict[userid]] = []  # 用户的id用0标识
for song_id in all_song_id:
    graph[songid_index_dict[song_id]] = []  # 歌曲的id用1标识

# 从用户听歌日志中将用户和歌曲的点击关系加入图中
user_song_list = train_data[['msno', 'song_id']].values.tolist()
for user_song in user_song_list:
    graph[userid_index_dict[user_song[0]]].append(songid_index_dict[user_song[1]])
    graph[songid_index_dict[user_song[1]]].append(userid_index_dict[user_song[0]])

with open(os.path.join(basedir, 'cache/graph.pkl'), 'wb') as f:
    pickle.dump(graph, f)