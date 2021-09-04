import math
from random import random
import numpy as np
import pdb
#对应run_pretrain_embeding  1
def sample_items_random(
        data,
        n_items,
        user_consumed_list,
        neg_label,
        num_neg=1
):
    user_consumed = {u: set(items) for u, items in user_consumed_list.items()}
    user_sampled = list()    
    item_sampled = list()
    label_sampled = list()
    for u, i in zip(data.user, data.item):
        #pdb.set_trace()            #看不懂，进行调试================================
        user_sampled.append(u)
        item_sampled.append(i)
        label_sampled.append(1.)
        for _ in range(num_neg):     #_为0或1
            item_neg = math.floor(n_items * random())         #随机选取负样本
            while item_neg in user_consumed[u]:
                item_neg = math.floor(n_items * random())
            user_sampled.append(u)            #user_sampled  [52158, 52158, 52158]
            item_sampled.append(item_neg)       #加入负样本    item_sampled   [102335, 597705]
            label_sampled.append(neg_label)      #负样本标签  该列表里只有两个值  label_sampled  [1.0, -1.0]
    return (
        np.array(user_sampled),
        np.array(item_sampled),
        np.array(label_sampled)
    )
