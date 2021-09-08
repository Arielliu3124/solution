import json
import os
import numpy as np


def save_npy(user_embeds, item_embeds, path):
    user_path = os.path.join(path, "user_embeddings_2.npy")
    item_path = os.path.join(path, "item_embeddings_2.npy")
    with open(user_path, "wb") as f:
        np.save(f, user_embeds)
    with open(item_path, "wb") as f:
        np.save(f, item_embeds)


def save_json(user_map, item_map, user_embeds, item_embeds, path):

    with open(os.path.join(path, "user_map_2.json"), "w") as f:
        json.dump(user_map, f, separators=(',', ':'))
    with open(os.path.join(path, "item_map_2.json"), "w") as f:
        json.dump(item_map, f, separators=(',', ':'))

