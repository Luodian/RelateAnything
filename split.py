import glob
import os 
import numpy as np
import json
from tqdm import tqdm

files = sorted(glob.glob('ram_data/feats/*'))

train = []
test = []

for npz in tqdm(files):
    tmp = np.load(npz,allow_pickle=True)
    rels = tmp['relations']
    for rel in rels:
        feat_id_1, feat_id_2, l = rel
        f1 = tmp['feat'][feat_id_1]
        f2 = tmp['feat'][feat_id_2]
        if f1 is not None and f2 is not None:
            if tmp['is_train']:
                train.append(npz.split('/')[-1])
            else:
                test.append(npz.split('/')[-1])
            break

with open('train_split.json', 'w') as f:
    json.dump(train, f)
with open('test_split.json', 'w') as f:
    json.dump(test, f)