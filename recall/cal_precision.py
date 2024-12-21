"""
评估召回算法的效果，计算命中率（hit rate）
"""
def hit_rate(recall_res, gt_res):
    """
    recall_res: {user_id1: set[recall_item1, recall_item2, ...], user_id2: set[recall_item1, ...], ...}
    gt_res: {user_id1: set[click_item1, click_item2, ...], user_id2: set[click_item1, ...], ...}

    命中率计算gt_res的所有N个点击记录中有多少被召回层召回
    """

    all_num = 0
    hit_num = 0
    for user_id, gt_item in gt_res.items():
        all_num += len(gt_item)
        for ite in gt_item:
            if ite in recall_res[user_id]:
                hit_num += 1

    return hit_num / all_num


import pandas as pd
import pickle
import os.path as osp

def get_gt_res(test_data_path, basedir='/Users/zhanghaoyang04/codeSpace/music_rec_system'):
    test_data = pd.read_csv(test_data_path)

    test_data = test_data.groupby('msno')['song_id'].apply(set).to_dict()

    pickle.dump(test_data, open(osp.join(basedir, 'cache/recall_gt_res.pkl'), 'wb'))

if __name__ == '__main__':
    get_gt_res('/Users/zhanghaoyang04/codeSpace/music_rec_system/data_for_test/test.csv')