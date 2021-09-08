import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import numpy as np
import pandas as pd
import pdb

# nohup python pre_data_2.py > test.log 2>&1 &
def parse_args(): 
    parser = argparse.ArgumentParser(description="pre_data_2")   #建立解析对象
    parser.add_argument("--seed", type=int, default=0)    #增加一个属性，--可将其变为可选参数
    return parser.parse_args()   #属性给与args实例：把parser中设置的所有"add_argument"给返回到args子类实例当中，那么parser中增加的属性内容都会在args实例中


def bucket_label(label):
    if label < 500:
        return 1
    elif label < 1000:
        return 2
    elif label < 2000:
        return 3
    else:
        return 4


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    np.random.seed(args.seed)
    start_time = time.perf_counter()

    user_feat = pd.read_csv("/home/math-tr/xq/solution/bigdata2021-rl-recsys/user_protrait_vec_2.csv", header=None,       #206096
                            names=["user", "user_p0", "user_p1", "user_p2","user_p3","user_p4","user_p5",
                            "user_p6","user_p7","user_p8","user_p9"])
    item_feat = pd.read_csv("/home/math-tr/xq/solution/bigdata2021-rl-recsys/item_info_vec.csv", header=None,
                            names=["item", "label", "location", "item_v0","item_v1","item_v2","item_v3","item_v4"])
    behavior = pd.read_csv("/home/math-tr/xq/solution/bigdata2021-rl-recsys/track2_testset_history.csv", header=None,
                           names=["user", "item", "time"])

    item_feat= item_feat.drop(index = (item_feat.loc[(item_feat['item_v4']==0)].index))
    user_counts = behavior.groupby("user")[["user"]].count().rename(
        columns={"user": "count_user"}
    ).sort_values("count_user", ascending=False)


    # sample users with short and long sequences
    item_counts = behavior.groupby("item")[["item"]].count().rename(
        columns={"item": "count_item"}
    ).sort_values("count_item", ascending=False)

    
    short_users = np.array(
        user_counts[
            (user_counts.count_user > -1) & (user_counts.count_user <= 3)
        ].index
    )
    print(len(short_users))
    for user in short_users:

        print(len(behavior))

        a1 = pd.Series({'user':user,'item':1,'time':1580572808})
        a2 = pd.Series({'user':user,'item':28,'time':1580572808})
        a3 = pd.Series({'user':user,'item':164,'time':1580572808})
        a4 = pd.Series({'user':user,'item':14,'time':1580572808})
        behavior = behavior.append(a1,ignore_index=True).append(a2,ignore_index=True).append(a3,ignore_index=True).append(a4,ignore_index=True)

    print(f"n_users: {behavior.user.nunique()}, "
          f"n_items: {behavior.item.nunique()}, "
          f"behavior length: {len(behavior)}")
 
    behavior = behavior.merge(user_feat, on="user")
    behavior = behavior.merge(item_feat, on="item")
    item_feat["label"] = item_feat["label"].apply(bucket_label)
    behavior = behavior.sort_values(by="time").reset_index(drop=True)

    behavior.to_csv("/home/math-tr/xq/solution/bigdata2021-rl-recsys/user_behavior_2.csv", header=None, index=False)
    print(f"prepare data done!, "
          f"time elapsed: {(time.perf_counter() - start_time):.2f}")



