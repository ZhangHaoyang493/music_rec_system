"""
给定的user_id和item_id是字符串，我们映射为整数，并使用字典存储，
之后使用pickle存储到本地
"""

import pickle
import pandas as pd
import os

user_info = pd.read_csv('../../data_for_test_scripts/members.csv')
user_id_list = user_info['msno'].unique().tolist()
# user_id: num
user_id_dict = dict(zip(user_id_list, range(len(user_id_list))))
# num: user_id
id_user_dict = dict(zip(range(len(user_id_list)), user_id_list))


songs_info = pd.read_csv('../../data_for_test_scripts/songs.csv')
song_id_list = songs_info['song_id'].unique().tolist()
# song_id: num
song_id_dict = dict(zip(song_id_list, range(len(song_id_list))))
# num: song_id
id_song_dict = dict(zip(range(len(song_id_list)), song_id_list))

if not os.path.exists('../cache'):
    os.mkdir('../cache')

# 测试一下
print(id_user_dict[0], user_id_dict[id_user_dict[0]])
print(id_song_dict[0], song_id_dict[id_song_dict[0]])

# 保存生成的字典
with open('../../cache/user_id_dict.pkl', 'wb') as f:
    pickle.dump(user_id_dict, f)

with open('../../cache/song_id_dict.pkl', 'wb') as f:
    pickle.dump(song_id_dict, f)

with open('../../cache/id_user_dict.pkl', 'wb') as f:
    pickle.dump(id_user_dict, f)

with open('../../cache/id_song_dict.pkl', 'wb') as f:
    pickle.dump(id_song_dict, f)