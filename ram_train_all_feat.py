import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import segment_anything
from torch.utils.data import Dataset,DataLoader
import json
import logging
import os

# Set up the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.binary_cross_entropy(F.sigmoid(inputs), targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class selfMLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, nhead=8, dropout=0.1,cls_qk_size=256):
        super().__init__()
        #Cross attention layer for subject and object
        # self.proj = nn.Linear(input_size,input_size)
        self.attn = nn.MultiheadAttention(input_size, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, input_size)

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
        # # Implementation of classifier
        # self.mlp = MLP(input_size*2, hidden_size, num_classes)
        self.cls_qk_size = cls_qk_size
        self.num_cls = num_classes
        self.cls_q = nn.Linear(input_size, self.num_cls * self.cls_qk_size)
        self.cls_k = nn.Linear(input_size, self.num_cls * self.cls_qk_size)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, feat,
                    attn_mask=None,
                    key_padding_mask=None,
                    sub_pos=None,
                    obj_pos=None):
        # Input query shape: B,N,d -> N,B,d
        B,N,d = feat.shape
        feat = feat.transpose(1,0)

        feat2 = self.attn(query=self.with_pos_embed(feat, sub_pos),
                                   key=self.with_pos_embed(feat, obj_pos),
                                   value=feat, attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask)[0]
        feat = feat + self.dropout1(feat2)
        feat = self.norm1(feat)
        feat2 = self.linear2(self.dropout(F.relu(self.linear1(feat))))
        feat = feat + self.dropout2(feat2)
        feat = self.norm2(feat).transpose(1,0) # N,B,d -> B,N,d

        q_embedding = self.cls_q(feat).reshape([B, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        k_embedding = self.cls_k(feat).reshape([B, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        cls_pred = q_embedding @ torch.transpose(k_embedding, 2, 3) / self.cls_qk_size ** 0.5 # B, cls, N, N

        return cls_pred


class fileset(Dataset):
    def __init__(self,split_file,dir):
        self.dir = dir
        with open(split_file, 'r') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.dir+self.data[idx]

class indexset(Dataset):
    def __init__(self,num):
        self.num = num
    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        return idx

def batch_pack(npz_name_list,device):
    B = len(npz_name_list)
    npz_list = [np.load(npz_name,allow_pickle=True) for npz_name in npz_name_list]

    max_len = max([npz['relations'].shape[0] for npz in npz_list])
    feat1 = []
    feat2 = []
    label = []
    padding_mask = torch.ones(B,max_len,dtype=torch.bool)
    for i in range(B):
        rels = npz_list[i]['relations']
        cur_feat1 = []
        cur_feat2 = []
        cur_label = []
        for rel in rels:
            feat_id_1, feat_id_2, l = rel
            f1 = npz_list[i]['feat'][feat_id_1]
            f2 = npz_list[i]['feat'][feat_id_2]
            if f1 is not None and f2 is not None:
                cur_feat1.append(torch.tensor(f1, dtype=torch.float32).unsqueeze(0))
                cur_feat2.append(torch.tensor(f2, dtype=torch.float32).unsqueeze(0))
                cur_label.append(torch.tensor(l, dtype=torch.long).unsqueeze(0))
        #Mark the valid pairs of a single image as True
        padding_mask[i,:len(cur_feat1)]=False
        feat1.append(torch.cat((torch.cat(cur_feat1), 1e-6*torch.ones((max_len-len(cur_label),256),dtype=torch.float32))).unsqueeze(0))
        feat2.append(torch.cat((torch.cat(cur_feat2), 1e-6*torch.ones((max_len-len(cur_label),256),dtype=torch.float32))).unsqueeze(0))
        label.append(torch.cat((torch.cat(cur_label), -1*torch.ones((max_len-len(cur_label)),dtype=torch.long))).unsqueeze(0))
    
    return torch.cat(feat1).to(device),torch.cat(feat2).to(device),torch.cat(label).to(device), padding_mask.to(device) #B,max_len,256 // B,max_len,256 // B,max_len // B,max_len


def multiclass_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    total = y_true.size(0)
    correct = (predicted == y_true).sum().item()
    return correct / total


def multiclass_mean_accuracy(output, labels):
    _, predicted = torch.max(output, 1)
    correct = (predicted == labels).squeeze()
    per_class_accuracy = torch.zeros(output.shape[1], device=device)

    for i in range(output.shape[1]):
        label_mask = (labels == i)
        if torch.sum(label_mask) > 0:
            per_class_accuracy[i] = torch.sum(correct & label_mask) / torch.sum(label_mask)

    return torch.mean(per_class_accuracy)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the MLP
input_size = 256
hidden_size = 512
num_classes = 56
batch_size = 128

learning_rate = 0.0001
num_epochs=1000

# Data
## Data option2
train_npz = np.load('train_rels_feat.npz',allow_pickle=True)
test_npz = np.load('test_rels_feat.npz',allow_pickle=True)
all_train_feat, all_train_feat_masks, all_train_rel_masks = \
    torch.tensor(train_npz['feat'], dtype=torch.float32), \
        torch.tensor(train_npz['feat_masks'], dtype=torch.bool), \
                torch.tensor(train_npz['rel_masks'], dtype=torch.bool)

trainset = indexset(all_train_feat_masks.shape[0])
train_index_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True)

test_feat, test_feat_masks, test_rel_masks = \
    torch.tensor(test_npz['feat'], dtype=torch.float32).to(device), \
        torch.tensor(test_npz['feat_masks'], dtype=torch.bool).to(device), \
                torch.tensor(test_npz['rel_masks'], dtype=torch.bool)


npzfile = np.load('ram_data/relation_feat.npz')
train_labels = npzfile['train_label']

# Create the model and move it to GPU
model = selfMLP(input_size, hidden_size, num_classes).to(device)

# Create the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

# Calculate class weights
class_counts = np.bincount(train_labels)
# class_weights = torch.tensor(np.sum(class_counts) / class_counts, dtype=torch.float32, device=device)

class_weights = torch.tensor(class_counts ** (1 / 10), dtype=torch.float32, device=device)
class_weights = class_weights / torch.sum(class_weights) *1000

# cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights).to(device)
cross_entropy_loss = nn.BCEWithLogitsLoss(weight=class_weights).to(device)
# cross_entropy_loss = nn.CrossEntropyLoss().to(device)

def one_hot_encoding(labels, num_classes):
    one_hot_labels = torch.zeros(labels.size(0), num_classes).to(device)
    one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
    return one_hot_labels.to(device)


# add focal loss 
focal_loss = FocalLoss().to(device)
focal_weight = 0.5

best_test_accuracy = 0
best_test_mean_accuracy = 0
# Training loop
for epoch in range(num_epochs):
    # ## For Data option 1
    # loss_train,train_accuracy=train_loop(trainloader, model, cross_entropy_loss, focal_weight, focal_loss, optimizer, device, batch_size, num_classes)
    # loss_test,test_accuracy,test_mean_accuracy=test_loop(testloader, model, cross_entropy_loss,device,num_classes)
    # logging.info(f"Epoch [{epoch+1}/{num_epochs}], "
    #     f"Train Loss: {loss_train:.4f}, Train Accuracy: {train_accuracy:.4f}, "
    #     f"Test Loss: {loss_test:.4f}, Test Accuracy: {test_accuracy:.4f}, "
    #     f"Test Mean Accuracy: {test_mean_accuracy:.4f}")
    ## For Data option 2
    # ### Manual shuffle
    # perm = torch.randperm(train_feat1.size(0))
    # train_feat1, train_feat2, train_label, train_mask = all_train_feat1[perm], all_train_feat2[perm], all_train_label[perm], all_train_mask[perm]
    
    ### Auto shuffle
    all_train_loss = 0
    all_output_train = None
    all_labels = None
    for batch, batch_index in enumerate(train_index_loader):
        model.train()
        train_feat, train_feat_mask, train_rel_mask = \
            all_train_feat[batch_index].to(device), all_train_feat_masks[batch_index].to(device), all_train_rel_masks[batch_index]
        # Forward pass
        output_train = model(train_feat,key_padding_mask=train_feat_mask)#B,cls,N,N

        # Filter the padding labels
        # mask =  #B,90,90,56 ->B,90,90 ->B*90*90 to filter the exist relations
        output_train = output_train.permute([0,2,3,1]).flatten(0,2)[train_rel_mask.sum(-1).to(torch.bool).flatten().to(device)] #B,56,90,90->B,90,90,56->B*90*90,56->valid_triplets,56
        multi_hot_label = train_rel_mask.flatten(0,2)[train_rel_mask.sum(-1).to(torch.bool).flatten()].to(device).to(torch.float32)##can be optimized
        label = multi_hot_label.to(torch.long).argmax(-1)
        # Calculate training loss
        # one_hot_label = one_hot_encoding(label, num_classes)
        loss_train_ce = cross_entropy_loss(output_train, multi_hot_label)
        loss_train_focal = focal_loss(output_train, multi_hot_label)

        loss_train = loss_train_ce + focal_weight * loss_train_focal


        # Backward pass
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        all_train_loss += loss_train.item()
        if all_output_train is None:
            all_output_train = output_train.cpu()
            all_labels = label.cpu()
        else:
            all_output_train =  torch.cat((all_output_train,output_train.cpu()))
            all_labels = torch.cat((all_labels,label.cpu()))
        if batch % 100 == 0:
            loss, current = loss_train.item(), (batch + 1) * batch_size
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{len(trainset):>5d}]")

    all_train_loss /= len(train_index_loader)
    # Calculate accuracy on the training set
    train_accuracy = multiclass_accuracy(all_output_train, all_labels)

    # Evaluate on the test set
    with torch.no_grad():
        model.eval()
        # Forward pass
        output_test = model(test_feat,key_padding_mask=test_feat_masks)#B,cls,N,N

        # Filter the padding labels
       #B,90,90,56 ->B,90,90 ->B*90*90 to filter the exist relations
        output_test = output_test.permute([0,2,3,1]).flatten(0,2)[test_rel_masks.sum(-1).to(torch.bool).flatten().to(device)] #B,56,90,90->B,90,90,56->B*90*90,56->valid_triplets,56
        multi_hot_labels = test_rel_masks.flatten(0,2)[test_rel_masks.sum(-1).to(torch.bool).flatten()].to(device).to(torch.float32)##can be optimized
        label = multi_hot_labels.to(torch.long).argmax(-1)

        # In the validation loop:
        # one_hot_test_labels = one_hot_encoding(label, num_classes)
        loss_test = cross_entropy_loss(output_test, multi_hot_labels)

        # Calculate accuracy on the test set
        test_accuracy = multiclass_accuracy(output_test, label)
         # Calculate accuracy on the test set
        test_mean_accuracy = multiclass_mean_accuracy(output_test, label)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {all_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Test Loss: {loss_test.item():.4f}, Test Accuracy: {test_accuracy:.4f}, "
          f"Test Mean Accuracy: {test_mean_accuracy:.4f}")

    # Inside the epoch loop, after calculating test_mean_accuracy
    if test_mean_accuracy > best_test_mean_accuracy:
        best_test_mean_accuracy = test_mean_accuracy
        torch.save(model.state_dict(), "best_mean_model_bce.pth")
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), "best_model_bce.pth")

# save model
logging.info(f'saved {best_test_accuracy:.4f}')
logging.info(f'saved {best_test_mean_accuracy:.4f}')
