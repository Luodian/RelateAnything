import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


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


def show_anns(anns, color_code='auto'):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        if color_code == 'auto':
            for i in range(3):
                img[:,:,i] = color_mask[i]
        elif color_code == 'red':
            for i in range(3):
                img[:,:,0] = 1
                img[:,:,1] = 0
                img[:,:,2] = 0
        else:
            for i in range(3):
                img[:,:,0] = 0
                img[:,:,1] = 0
                img[:,:,2] = 1
    return np.dstack((img, m*0.35))


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)   

def show_mask(m):
    img = np.ones((m.shape[0], m.shape[1], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        img[:,:,0] = 1
        img[:,:,1] = 0
        img[:,:,2] = 0

    return np.dstack((img, m*0.35))


def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def sort_and_deduplicate(sam_masks, iou_threshold=0.8):
    # Sort the sam_masks list based on the area value
    sorted_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)

    # Deduplicate masks based on the given iou_threshold
    filtered_masks = []
    for mask in sorted_masks:
        duplicate = False
        for filtered_mask in filtered_masks:
            if iou(mask['segmentation'], filtered_mask['segmentation']) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            filtered_masks.append(mask)

    return filtered_masks


relation_classes = ['over',
        'in front of',
        'beside',
        'on',
        'in',
        'attached to',
        'hanging from',
        'on back of',
        'falling off',
        'going down',
        'painted on',
        'walking on',
        'running on',
        'crossing',
        'standing on',
        'lying on',
        'sitting on',
        'flying over',
        'jumping over',
        'jumping from',
        'wearing',
        'holding',
        'carrying',
        'looking at',
        'guiding',
        'kissing',
        'eating',
        'drinking',
        'feeding',
        'biting',
        'catching',
        'picking',
        'playing with',
        'chasing',
        'climbing',
        'cleaning',
        'playing',
        'touching',
        'pushing',
        'pulling',
        'opening',
        'cooking',
        'talking to',
        'throwing',
        'slicing',
        'driving',
        'riding',
        'parked on',
        'driving on',
        'about to hit',
        'kicking',
        'swinging',
        'entering',
        'exiting',
        'enclosing',
        'leaning on',]
relation_classes = [
    'over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of', 'falling off', 
    'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on', 'lying on', 'sitting on', 
    'flying over', 'jumping over', 'jumping from', 'wearing', 'holding', 'carrying', 'looking at', 'guiding', 
    'kissing', 'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing', 
    'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening', 'cooking', 'talking to', 
    'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging', 
    'entering', 'exiting', 'enclosing', 'leaning on']

from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


input_size = 512
hidden_size = 256
num_classes = 56

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

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def sort_and_deduplicate(sam_masks, iou_threshold=0.8):
    # Sort the sam_masks list based on the area value
    sorted_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)

    # Deduplicate masks based on the given iou_threshold
    filtered_masks = []
    for mask in sorted_masks:
        duplicate = False
        for filtered_mask in filtered_masks:
            if iou(mask['segmentation'], filtered_mask['segmentation']) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            filtered_masks.append(mask)

    return filtered_masks

def show_anns(anns, color_code='auto'):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        if color_code == 'auto':
            for i in range(3):
                img[:,:,i] = color_mask[i]
        elif color_code == 'red':
            for i in range(3):
                img[:,:,0] = 1
                img[:,:,1] = 0
                img[:,:,2] = 0
        else:
            for i in range(3):
                img[:,:,0] = 0
                img[:,:,1] = 0
                img[:,:,2] = 1
    return np.dstack((img, m*0.35))

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)   

def show_mask(m):
    img = np.ones((m.shape[0], m.shape[1], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        img[:,:,0] = 1
        img[:,:,1] = 0
        img[:,:,2] = 0

    return np.dstack((img, m*0.35))