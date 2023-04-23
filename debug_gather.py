import numpy as np
import json
from tqdm import tqdm

train_file = 'train_split.json'
test_file = 'test_split.json'
with open(train_file, 'r') as f:
    train_list = json.load(f)
with open(test_file, 'r') as f:
    test_list = json.load(f)
    
train_num_rels = []#max=41 padding to 50
num_feat = []#max=81 padding to 90

train_feat = []
train_rel_masks = []
for file in tqdm(train_list):
    with np.load('ram_data/feats_0420/'+file, allow_pickle=True) as data:
        cur_feat = []
        index_map = {}
        for i in range(len(data['feat'])):
            f = data['feat'][i]
            if f is not None:
                cur_feat.append(np.array(f)[np.newaxis,:])
                index_map[i]=len(cur_feat)-1
        rel_mask = np.zeros((1,90,90,56),dtype=bool)
        for rel in data['relations']:
            s,o,r = rel
            if s in index_map and o in index_map:
                rel_mask[0,index_map[s],index_map[o],r]=True
        train_feat.append(np.concatenate(cur_feat)[np.newaxis,:])#1,n,256
        train_rel_masks.append(rel_mask)#1,90,90,56
        assert train_feat[-1].ndim == 3 
        assert train_rel_masks[-1].ndim == 4

test_feat = []
test_rel_masks = []
for file in tqdm(test_list):
    with np.load('ram_data/feats_0420/'+file, allow_pickle=True) as data:
        cur_feat = []
        index_map = {}
        for i in range(len(data['feat'])):
            f = data['feat'][i]
            if f is not None:
                cur_feat.append(np.array(f)[np.newaxis,:])
                index_map[i]=len(cur_feat)-1
        rel_mask = np.zeros((1,90,90,56),dtype=bool)
        for rel in data['relations']:
            s,o,r = rel
            if s in index_map and o in index_map:
                rel_mask[0,index_map[s],index_map[o],r]=True
        test_feat.append(np.concatenate(cur_feat)[np.newaxis,:])#1,n,256
        test_rel_masks.append(rel_mask)#1,90,90,56
        assert test_feat[-1].ndim == 3 
        assert test_rel_masks[-1].ndim==4

feat_masks = []
for i in tqdm(range(len(train_feat))):
    feat_mask = np.ones((1,90),dtype=bool)

    cur_objs = train_feat[i].shape[1]
    train_feat[i] = np.concatenate((train_feat[i],1e-6*np.ones((1,90-cur_objs,256))),axis=1)
    feat_mask[:,:cur_objs]=False
    feat_masks.append(feat_mask)

    assert train_feat[i].shape[1]==feat_masks[i].shape[1]==90

test_feat_masks = []
for i in tqdm(range(len(test_feat))):
    test_feat_mask = np.ones((1,90),dtype=bool)

    cur_objs = test_feat[i].shape[1]
    test_feat[i] = np.concatenate((test_feat[i],1e-6*np.ones((1,90-cur_objs,256))),axis=1)
    test_feat_mask[:,:cur_objs]=False
    test_feat_masks.append(test_feat_mask)

    assert test_feat[i].shape[1]==test_feat_masks[i].shape[1]==90

train_feat = np.concatenate(train_feat)
train_rel_masks = np.concatenate(train_rel_masks)
feat_masks = np.concatenate(feat_masks)
assert train_feat.shape[0]==train_rel_masks.shape[0]==feat_masks.shape[0]

test_feat = np.concatenate(test_feat)
test_rel_masks = np.concatenate(test_rel_masks)
test_feat_masks = np.concatenate(test_feat_masks)
assert test_feat.shape[0]==test_rel_masks.shape[0]==test_feat_masks.shape[0]
np.savez('train_rels_feat_0420', 
         feat=train_feat, # Size,90,256
         feat_masks=feat_masks, # Size,90
         rel_masks=train_rel_masks) # Size,90,90,56
np.savez('test_rels_feat_0420', 
         feat=test_feat, 
         feat_masks=test_feat_masks,
         rel_masks=test_rel_masks)