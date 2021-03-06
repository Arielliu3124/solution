import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from solution.data import process_feat_data, FeatDataset
from solution.models import DSSM
from solution.utils import sample_items_random, init_param_dssm, generate_embeddings
from solution.trainer import pretrain_model
from solution.serialization import save_npy, save_json
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="run_pretrain_embeddings_kaggle")
    parser.add_argument("--data", type=str, default="user_behavior_2.csv")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--loss", type=str, default="cosine",
                        help="cosine or bce loss")
    parser.add_argument("--neg_item", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("A list all args: \n======================")
    pprint(vars(args))
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    PATH = os.path.join("bigdata2021-rl-recsys", args.data)
    EMBEDDING_PATH = "bigdata2021-rl-recsys/"

    static_feat = ["user_p0","user_p1", "user_p2","user_p3","user_p4","user_p5",
                         "user_p6","user_p7","user_p8","user_p9"]
    dynamic_feat = ["item_v0","item_v1","item_v2","item_v3","item_v4"]
    # static_feat =None
    # dynamic_feat =None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    item_embed_size = args.embed_size
    feat_embed_size = args.embed_size
    hidden_size = (64, 128)
    criterion = (
        nn.CosineEmbeddingLoss()
        if args.loss == "cosine"
        else nn.BCEWithLogitsLoss()                #?????????????????????????????????
    )
    criterion_type = (
        "cosine"
        if "cosine" in criterion.__class__.__name__.lower()
        else "bce"
    )
    neg_label = -1. if criterion_type == "cosine" else 0.
    neg_item = args.neg_item                  
    
    columns = ["user", "item", "time","user_p0", "user_p1", "user_p2","user_p3","user_p4","user_p5",
                            "user_p6","user_p7","user_p8","user_p9", "label", "location", "item_v0","item_v1","item_v2","item_v3","item_v4"]
    #????????????  ?????????????????? process_feat_data
    # (n_users,n_items,train_user_consumed,eval_user_consumed,train_data,eval_data,user_map,item_map,feat_map) = process_feat_data(PATH, 
    # columns, test_size=0.2, time_col="time",static_feat=static_feat, dynamic_feat=dynamic_feat)
    (n_users,n_items,train_user_consumed,eval_user_consumed,train_data,eval_data,user_map,item_map,feat_map) = process_feat_data(PATH, 
    columns, test_size=0.2, time_col="time",static_feat=static_feat, dynamic_feat=dynamic_feat)
    print(f"n_users: {n_users}, n_items: {n_items}, "
          f"train_shape: {train_data.shape}, eval_shape: {eval_data.shape}")

    train_user, train_item, train_label = sample_items_random(
        train_data, n_items, train_user_consumed, neg_label, neg_item
    )         #???utils??????
    eval_user, eval_item, eval_label = sample_items_random(
        eval_data, n_items, eval_user_consumed, neg_label, neg_item
    )

    train_dataset = FeatDataset(
        train_user,
        train_item,
        train_label,
        feat_map,
        static_feat,
        dynamic_feat
    )
    eval_dataset = FeatDataset(
        eval_user,
        eval_item,
        eval_label,
        feat_map,
        static_feat,
        dynamic_feat
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    
    model = DSSM(
        item_embed_size,
        feat_embed_size,
        n_users,
        n_items,
        hidden_size,
        feat_map,
        static_feat,
        dynamic_feat,
        use_bn=True
    ).to(device)
    init_param_dssm(model)
    optimizer = Adam(model.parameters(), lr=lr)  # weight_decay

    pretrain_model(model, train_loader, eval_loader, n_epochs, criterion,
                   criterion_type, optimizer, device)
    user_embeddings, item_embeddings = generate_embeddings(
        model, n_users, n_items, feat_map, static_feat, dynamic_feat, device
    ) 
    #pdb.set_trace()            
    print(f"user_embeds shape: {user_embeddings.shape},"
          f" item_embeds shape: {item_embeddings.shape}")

    save_npy(user_embeddings, item_embeddings, EMBEDDING_PATH)
    save_json(
        user_map, item_map, user_embeddings, item_embeddings, EMBEDDING_PATH
    )
    print("pretrain embeddings done!")
