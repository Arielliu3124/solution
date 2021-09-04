import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
from pprint import pprint
import numpy as np
import torch
from torch.optim import Adam
from dbrl_1.data import process_data, build_dataloader
from dbrl_1.models import DDPG
from dbrl_1.network import Actor, Critic
from dbrl_1.trainer import train_model
from dbrl_1.utils.params import count_vars, init_param
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="run_ddpg_kaggle")
    parser.add_argument("--data", type=str, default="user_behavior_2.csv")
    parser.add_argument("--user_embeds", type=str,
                        default="user_embeddings_2.npy")
    parser.add_argument("--item_embeds", type=str,
                        default="item_embeddings_2.npy")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--hist_num", type=int, default=10,
                        help="num of history items to consider")
    parser.add_argument("--n_rec", type=int, default=10,
                        help="num of items to recommend")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--policy_delay", type=int, default=1)
    parser.add_argument("--sess_mode", type=str, default="one",
                        help="Specify when to end a session")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("A list all args: \n======================")
    pprint(vars(args))
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH = os.path.join("bigdata2021-rl-recsys", args.data)
    with open(os.path.join("bigdata2021-rl-recsys", args.user_embeds), "rb") as f:
        user_embeddings = np.load(f)
    with open(os.path.join("bigdata2021-rl-recsys", args.item_embeds), "rb") as f:
        item_embeddings = np.load(f)
    item_embeddings[-1] = 0.   # last item is used for padding
    pdb.set_trace()
    n_epochs = args.n_epochs
    hist_num = args.hist_num
    batch_size = eval_batch_size = args.batch_size
    embed_size = item_embeddings.shape[1]
    #pdb.set_trace()                                 #pdb
    hidden_size = args.hidden_size
    input_dim = embed_size * (hist_num + 1)
    action_dim = embed_size
    actor_lr = args.lr
    critic_lr = args.lr
    weight_decay_actor = args.weight_decay
    weight_decay_critic = args.weight_decay
    gamma = args.gamma
    tau = args.tau
    n_rec = args.n_rec    #推荐的数量
    policy_delay = args.policy_delay
    pad_val = len(item_embeddings) - 1     #不知道是做什么的
    sess_mode = args.sess_mode
    neg_sample = None
    one_hour = int(60 * 60)
    #reward_map = {"pv": 1., "cart": 2., "fav": 2., "buy": 3.}
    reward_map = {"1": 1., "2": 2., "3": 3., "4": 4.}
    '''
    columns = ["user", "item", "label", "time", "sex", "age", "pur_power",
               "category", "shop", "brand"]'''
    columns = ["user", "item", "time","user_p0", "user_p1", "user_p2","user_p3","user_p4","user_p5",
                            "user_p6","user_p7","user_p8","user_p9", "label", "location", "item_v0","item_v1","item_v2","item_v3","item_v4"]

#转到data中  process_data
    (
        n_users,
        n_items,
        train_user_consumed,
        test_user_consumed,
        train_sess_end,
        test_sess_end,
        train_rewards,
        test_rewards
    ) = process_data(PATH, columns, 0.2, time_col="time", sess_mode=sess_mode,
                     interval=one_hour, reward_shape=reward_map)
    #转到data中
    train_loader, eval_loader = build_dataloader(
        n_users,
        n_items,
        hist_num,
        train_user_consumed,
        test_user_consumed,
        batch_size,
        sess_mode=sess_mode,
        train_sess_end=train_sess_end,
        test_sess_end=test_sess_end,
        n_workers=0,
        neg_sample=neg_sample,
        train_rewards=train_rewards,
        test_rewards=test_rewards,
        reward_shape=reward_map
    )

#跳转到network
    actor = Actor(
        input_dim, action_dim, hidden_size, user_embeddings, item_embeddings,
        None, pad_val, 1, device).to(device)
    critic = Critic(input_dim, action_dim, hidden_size).to(device)
    #在utils中   参数初始化
    init_param(actor, critic)

    actor_optim = Adam(
        actor.parameters(), actor_lr, weight_decay=weight_decay_actor
    )
    critic_optim = Adam(
        critic.parameters(), critic_lr, weight_decay=weight_decay_critic
    )
    #跳转到modals 中的DDPG
    model = DDPG(
        actor,
        actor_optim,
        critic,
        critic_optim,
        tau,
        gamma,
        policy_delay=policy_delay,
        item_embeds=item_embeddings,
        device=device
    )

    var_counts = tuple(count_vars(module) for module in [actor, critic])
    print(f'Number of parameters: actor: {var_counts[0]}, '
          f' critic: {var_counts[1]}')

    train_model(
        model,
        n_epochs,
        n_rec,
        n_users,
        train_user_consumed,
        test_user_consumed,
        hist_num,
        train_loader,
        eval_loader,
        item_embeddings,
        eval_batch_size,
        pad_val,
        device,
        debug=True,
        eval_interval=10
    )

    torch.save(actor.state_dict(), "bigdata2021-rl-recsys/model_ddpg.pt")
    print("train and save done!")
