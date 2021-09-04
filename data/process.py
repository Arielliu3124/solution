from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from .session import build_sess_end
from .split import split_by_ratio
import pdb

#run_ddpg  1
'''
def process_data(path, columns=None, test_size=0.2, time_col="time",
                 sess_mode="one", interval=None, reward_shape=None):
 '''
def process_data(path, columns=None, test_size=0.2, time_col="time",
                 sess_mode="one", interval=None, reward_shape=None):
    """Split and process data before building dataloader.

    Parameters
    ----------
    path : str
        File path.
    columns : list, optional
        Column names for the original data.
    test_size : float
        Test data size to split from original data.
    time_col : str
        Specify which column represents time.
    sess_mode : str
        Ways of representing a session.
    interval : int
        Interval between different sessions.
    reward_shape : dict (default None)
        A dict for mapping labels to rewards.

    Returns
    -------
    n_users : int
        Number of users.
    n_items : int
        Number of items.
    train_user_consumed : dict
        Items interacted by each user in train data.
    test_user_consumed : dict
        Items interacted by each user in test data.
    train_sess_end : dict
        Session end mark for each user in train data.
    test_sess_end : dict
        Session end mark for each user in test data.
    train_rewards : dict (default None)
        A dict for mapping train users to rewards.
    test_rewards : dict (default None)
        A dict for mapping test users to rewards.
    """

    column_names = columns if columns is not None else None
    data = pd.read_csv(path, sep=",", names=column_names)
    #assert为断言语句
    assert time_col in data.columns, "must specify correct time column name..."

    # data = data.sort_values(by=time_col).reset_index(drop=True)
    #跳转到split_by_ratio
    train_data, test_data = split_by_ratio(data, shuffle=False,
                                           test_size=test_size,
                                           pad_unknown=True,
                                           seed=42)

    #跳转到 2
    train_data, test_data = map_unique_value(train_data, test_data)
    n_users = train_data.user.nunique()
    n_items = train_data.item.nunique()
    # pandas groupby too slow...
    #train_user_consumed = train_data.groupby("user")["item"].apply(
    #    lambda x: list(x)).to_dict()
    # test_user_consumed = test_data.groupby("user")["item"].apply(
    #    lambda x: list(x)).to_dict()
    #train_user_consumed = train_data.groupby("user")["item","price"].aggregate(lambda x: ','.join(map(str, x)))
    #train_user_consumed.to_csv("/home/math-tr/xq/dbrl_1/bigdata2021-rl-recsys/train_user_consumed.csv", header=None, index=False)
    #print(type(train_user_consumed))
    #pdb.set_trace()
    train_user_consumed = build_interaction(train_data)
    #pdb.set_trace()
    test_user_consumed = build_interaction(test_data)
    if reward_shape is not None:
        train_rewards = build_reward(train_data, reward_shape)
        test_rewards = build_reward(test_data, reward_shape)
    else:
        train_rewards = test_rewards = None

    train_sess_end = build_sess_end(train_data, sess_mode, time_col, interval)
    test_sess_end = build_sess_end(test_data, sess_mode, time_col, interval)

    result = (
        n_users,
        n_items,
        train_user_consumed,
        test_user_consumed,
        train_sess_end,
        test_sess_end,
        train_rewards,
        test_rewards
    )
    return result

#对应run_train_embeeding 1————2————3————4
def process_feat_data(path, columns=None, test_size=0.2, time_col="time",
                 static_feat=None, dynamic_feat=None):
    column_names = columns if columns is not None else None
    data = pd.read_csv(path, sep=",", names=column_names)
    # data = data.sort_values(by=time_col).reset_index(drop=True)
    train_data, test_data = split_by_ratio(data, shuffle=False,
                                           test_size=test_size,
                                           pad_unknown=True,
                                           seed=42)

    train_data, test_data, user_map, item_map = map_unique_value_feat(
        train_data, test_data, static_feat, dynamic_feat
    )
    n_users = train_data.user.nunique()
    n_items = train_data.item.nunique()
    train_user_consumed = build_interaction(train_data)  #3
    test_user_consumed = build_interaction(test_data)
    feat_map = build_feat_map(train_data, n_items, static_feat, dynamic_feat)    #4
    result = (
        n_users,
        n_items,
        train_user_consumed,
        test_user_consumed,
        train_data,
        test_data,
        user_map,
        item_map,
        feat_map
    )
    return result

#接上
def map_unique_value(train_data, test_data):   #这个函数没有看懂
    for col in ["user", "item"]:
        # unique_vals = np.unique(train_data[col])
        # mapping = dict(zip(unique_vals, range(len(unique_vals))))
        # map according to frequency
        counts = train_data[col].value_counts()    #显示col值
        freq = counts.index.tolist()    #tolist 将矩阵数组转换为列表
        mapping = dict(zip(freq, range(len(freq))))
        train_data[col] = train_data[col].map(mapping)       #map(function,iterable,...)，这个函数的意思就是将function应用于iterable的每一个元素，结果以列表的形式返回。
        test_data[col] = test_data[col].map(mapping)
        if test_data[col].isnull().any():
            col_type = train_data[col].dtype
            test_data[col].fillna(len(freq), inplace=True)
            test_data[col] = test_data[col].astype(col_type)
    return train_data, test_data

#对应run_train_embedding 2
def map_unique_value_feat(train_data, test_data, static_feat=None,
                          dynamic_feat=None):
    user_map = dict()          #{1972497: 79996, 1024254: 79997, 148905: 79998, 1201651: 79999}
    item_map = dict()
    total_cols = ["user", "item"]
    if static_feat is not None:
        total_cols.extend(static_feat)
    if dynamic_feat is not None:
        total_cols.extend(dynamic_feat)
    for col in total_cols:   #total_cols = ['user', 'item', 'sex', 'age', 'pur_power', 'category', 'shop', 'brand']
        # map according to frequency
        #pdb.set_trace()     #进行调试
        counts = train_data[col].value_counts()   #对col计数，并且从多到少排序  24354427 873  。。。。  9873807 1
        freq = counts.index.tolist()             # 对item   [24354427,...,25211930, 37800981, 26801195, 16792587, 9873807]
        mapping = dict(zip(freq, range(len(freq))))    # 对item  {24354427:1,...,26801195: 920799, 16792587: 920800, 9873807: 920801}
        train_data[col] = train_data[col].map(mapping)    
        '''train_data['item']
        102267     102335
        104219     138294
               ...
        2665901    310761
        Name: item, Length: 2595573, dtype: int64
        '''
        test_data[col] = test_data[col].map(mapping)
        if col == "user":
            user_map = mapping.copy()   #{1972497: 79996, 1024254: 79997, 148905: 79998, 1201651: 79999}
        elif col == "item":
            item_map = mapping.copy()    
        if test_data[col].isnull().any():
            col_type = train_data[col].dtype
            test_data[col].fillna(len(freq), inplace=True)
            test_data[col] = test_data[col].astype(col_type)
    return train_data, test_data, user_map, item_map

#对应run_train_embeeding 3
def build_interaction(data):
    consumed = defaultdict(list)
    for u, i in zip(data.user.tolist(), data.item.tolist()):
        consumed[u].append(i)
    return consumed

#接上
def build_reward(data, reward_shape):
    """
    Reward format: {user_index: {label_name: list of index in interaction}
    For example, {1: {"pv": [1,3,5,6]}, {"cart": [2]}, {"buy": [4]}}
    """
    label_all = defaultdict(list)
    for u, l in zip(data.user.tolist(), data.label.tolist()):
        label_all[u].append(l)

    reward_all = defaultdict(dict)
    for user, label in label_all.items():
        for key, rew in reward_shape.items():
            index = np.where(np.array(label) == key)[0]
            #pdb.set_trace()
            if len(index) > 0:
                reward_all[user].update({key: index})
    return reward_all

#对应run_train_embeeding 4
def build_feat_map(data, n_items, static_feat=None, dynamic_feat=None):
    feat_map = dict()
    if static_feat is not None:
        for feat in static_feat:
            feat_map[feat] = dict(zip(data["user"], data[feat]))
            feat_map[feat + "_vocab"] = data[feat].nunique()
    if dynamic_feat is not None:
        for feat in dynamic_feat:
            feat_map[feat] = dict(zip(data["item"], data[feat]))
            feat_map[feat + "_vocab"] = data[feat].nunique()
            # avoid oov item features
            feat_map[feat][n_items] = feat_map[feat + "_vocab"]
    return feat_map


# def build_batch_data(mode, train_user_consumed, hist_num, n_users, device):
#    if mode == "train":
#        items = [train_user_consumed[u][:hist_num] for u in range(n_users)]
#    else:
#        items = [train_user_consumed[u][-hist_num:] for u in range(n_users)]
#    data = {"user": torch.as_tensor(range(n_users)).to(device),
#            "item": torch.as_tensor(items).to(device)}
#    return data


def build_batch_data(mode, train_user_consumed, hist_num, n_users, pad_val,
                     device):
    items = np.array(
        [
            last_history(train_user_consumed[u], hist_num, pad_val, mode)
            for u in range(n_users)
        ]
    )
    data = {"user": torch.as_tensor(range(n_users)).to(device),
            "item": torch.from_numpy(items).to(device)}
    return data


def last_history(history, hist_num, pad_val, mode):
    hist_len = len(history)
    if hist_len < hist_num:
        rec_hist = np.full(hist_num, pad_val, dtype=np.int64)
        rec_hist[-hist_len:] = history
    else:
        rec_hist = (
            np.array(history[:hist_num])
            if mode == "train"
            else np.array(history[-hist_num:])
        )
    return rec_hist

