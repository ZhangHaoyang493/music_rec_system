import pandas as pd

# 如果用于正式的计算，就使用data
# 如果用于小批量的数据进行测试，就使用 data_for_test_scripts
path_name = 'data_for_test_scripts'

# 构造用户最喜欢的artist_name

# 定义函数，选出每个用户听的最多的五个genre_id值，不够的用-1填充
def favourite_generes(x):
    # value_counts()会自动忽略NaN值
    top_genres = x['genre_ids'].value_counts().head(5).index.tolist()
    top_genres += [-1] * (5 - len(top_genres))
    return ','.join(map(str, top_genres))

def favourite_artist_name(x):
    top_artist_name = x['artist_name'].value_counts().head(5).index.tolist()
    top_artist_name += [-1] * (5 - len(top_artist_name))
    return ','.join(map(str, top_artist_name))

def favourite_composer(x):
    top_composer = x['composer'].value_counts().head(5).index.tolist()
    top_composer += [-1] * (5 - len(top_composer))
    return ','.join(map(str, top_composer))

def favourite_lyricist(x):
    top_lyricist = x['lyricist'].value_counts().head(5).index.tolist()
    top_lyricist += [-1] * (5 - len(top_lyricist))
    return ','.join(map(str, top_lyricist))

def favourite_language(x):
    top_language = x['language'].value_counts().head(5).index.tolist()
    top_language += [-1] * (5 - len(top_language))
    return ','.join(map(str, top_language))


# 读取数据
train_data = pd.read_csv('../../%s/train.csv' % path_name, encoding='utf-8')
song_data = pd.read_csv('../../%s/songs.csv' % path_name, encoding='utf-8')

merge_data = pd.merge(train_data, song_data, on='song_id', how='left')

# 构建用户最喜欢的类型、喜欢的歌手等特征
user_favourite_genres = merge_data.groupby('msno').apply(favourite_generes).reset_index()
user_favourite_genres.columns = ['msno', 'favourite_top_5_genres']

user_favourite_artist_name = merge_data.groupby('msno').apply(favourite_artist_name).reset_index()
user_favourite_artist_name.columns = ['msno', 'favourite_top_5_artist_name']

user_favourite_composer = merge_data.groupby('msno').apply(favourite_composer).reset_index()
user_favourite_composer.columns = ['msno', 'favourite_top_5_composer']

user_favourite_lyricist = merge_data.groupby('msno').apply(favourite_lyricist).reset_index()
user_favourite_lyricist.columns = ['msno', 'favourite_top_5_lyricist']

user_favourite_language = merge_data.groupby('msno').apply(favourite_language).reset_index()
user_favourite_language.columns = ['msno', 'favourite_top_5_language']

# 将结果合并到merge_data中
merge_data = pd.merge(merge_data, user_favourite_genres, on='msno', how='left')
del user_favourite_genres
merge_data = pd.merge(merge_data, user_favourite_artist_name, on='msno', how='left')
del user_favourite_artist_name
merge_data = pd.merge(merge_data, user_favourite_composer, on='msno', how='left')
del user_favourite_composer
merge_data = pd.merge(merge_data, user_favourite_lyricist, on='msno', how='left')
del user_favourite_lyricist
merge_data = pd.merge(merge_data, user_favourite_language, on='msno', how='left')
del user_favourite_language

# 构建用户听歌的听歌长度的最大值，最小值，平均值，方差
user_song_length_max = merge_data.groupby('msno')['song_length'].max().reset_index()
user_song_length_max.columns = ['msno', 'user_song_length_max']
user_song_length_min = merge_data.groupby('msno')['song_length'].min().reset_index()
user_song_length_min.columns = ['msno', 'user_song_length_min']
user_song_length_mean = merge_data.groupby('msno')['song_length'].mean().reset_index()
user_song_length_mean.columns = ['msno', 'user_song_length_mean']
user_song_length_std = merge_data.groupby('msno')['song_length'].std().reset_index()
user_song_length_std.columns = ['msno', 'user_song_length_std']

# 将结果合并到merge_data中
merge_data = pd.merge(merge_data, user_song_length_max, on='msno', how='left')
del user_song_length_max
merge_data = pd.merge(merge_data, user_song_length_min, on='msno', how='left')
del user_song_length_min
merge_data = pd.merge(merge_data, user_song_length_mean, on='msno', how='left')
del user_song_length_mean
merge_data = pd.merge(merge_data, user_song_length_std, on='msno', how='left')
del user_song_length_std

# 构建用户的听歌历史特征，由于没有时间戳，就从用户听过的歌里面随机选取20个歌曲，如果不够20个，就用-1填充
# 优先选择用户target=1的歌曲，因为这些歌曲用户更喜欢，反复听了
# 再选择用户target=0的歌曲
# 如果用户没有听过任何歌曲，就用-1填充
def user_song_history(x):
    target_1_song = x[x['target'] == 1]['song_id'].tolist()
    target_0_song = x[x['target'] == 0]['song_id'].tolist()
    if len(target_1_song) >= 20:
        return ','.join(target_1_song[:20])
    else:
        target_1_song += target_0_song
        target_1_song += ['-1'] * (20 - len(target_1_song))
        return ','.join(target_1_song)
    
user_song_history = merge_data.groupby('msno').apply(user_song_history).reset_index()
user_song_history.columns = ['msno', 'user_song_history']
# 将结果合并到merge_data中
merge_data = pd.merge(merge_data, user_song_history, on='msno', how='left')


# 保存结果
merge_data_partial = merge_data[['msno', 
                                 'favourite_top_5_genres', 
                                 'favourite_top_5_artist_name', 
                                 'favourite_top_5_composer',
                                 'favourite_top_5_lyricist',
                                 'favourite_top_5_language',
                                 'user_song_length_max',
                                 'user_song_length_min',
                                 'user_song_length_mean',
                                 'user_song_length_std',
                                 'user_song_history']]


# 和用户的基本信息连接一并保存
user_basic_data = pd.read_csv('../../data_for_test_scripts/members.csv', encoding='utf-8')
merge_data_partial = pd.merge(user_basic_data, merge_data_partial, on='msno', how='left')
merge_data_partial.drop_duplicates(subset=['msno'], inplace=True)

merge_data_partial.to_csv('../../cache/user_feature_for_recall.csv', index=False)