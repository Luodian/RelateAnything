from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path
import matplotlib.pyplot as plt
import pprint
from tqdm import tqdm

import numpy as np
from PIL import Image

from openpsg.utils.vis_tools.datasets import coco_dir
from openpsg.utils.vis_tools.preprocess import load_json

from detectron2.data.detection_utils import read_image
from detectron2.utils.colormap import colormap
from panopticapi.utils import rgb2id

import argparse, os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='ID of the part to process')
    parser.add_argument('--start_id', type=int, help='ID to start the process')
    parser.add_argument('--num_parts', type=int, help='ID to start the process')
    parser.add_argument('--total_images', type=int, help='ID to start the process')

    args = parser.parse_args()
    return args

predictor = SamPredictor(
    build_sam(checkpoint="./segment_anything/checkpoints/sam_vit_h_4b8939.pth").to(device='cuda'))

mask_generator = SamAutomaticMaskGenerator(
    build_sam(checkpoint="./segment_anything/checkpoints/sam_vit_h_4b8939.pth").to(device='cuda'))

# set working path as home dir to easy access data
psg_dataset_file = load_json(Path("data/psg/psg.json"))
print('keys: ', list(psg_dataset_file.keys()))

psg_thing_cats = psg_dataset_file['thing_classes']
psg_stuff_cats = psg_dataset_file['stuff_classes']
psg_obj_cats = psg_thing_cats + psg_stuff_cats
psg_rel_cats = psg_dataset_file['predicate_classes']
psg_dataset = {d["image_id"]: d for d in psg_dataset_file['data']}
# psg_dataset_coco_id = {d["coco_image_id"]: d for d in psg_dataset_file['data']}

print('Number of images: {}'.format(len(psg_dataset)))
print('# Object Classes: {}'.format(len(psg_obj_cats)))
print('# Relation Classes: {}'.format(len(psg_rel_cats)))


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


def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def process_image(img_id):
    # get all data info
    data = psg_dataset[img_id]

    # show origin image
    image = read_image(coco_dir / data["file_name"], format="RGB")
    seg_map = read_image(coco_dir / data["pan_seg_file_name"], format="RGB")
    seg_map = rgb2id(seg_map)

    # get seperate masks
    gt_masks = []
    labels_coco = []
    for i, s in enumerate(data["segments_info"]):
        label = psg_obj_cats[s["category_id"]]
        labels_coco.append(label)
        gt_masks.append(seg_map == s["id"])

    # pil image loading for sam
    pilimage = Image.open(coco_dir / data["file_name"])
    width, height = pilimage.size
    full_area = width * height

    # seg anything
    sam_masks = mask_generator.generate(image)
    ddup_masks = sort_and_deduplicate(sam_masks)

    # GT matching
    gt_index_mgt = []
    for mask_id, mask_dict in enumerate(ddup_masks):
        max_iou = 0
        for gt_mask_id, gt_mask in enumerate(gt_masks):
            current_iou = iou(gt_mask, mask_dict['segmentation'])
            if current_iou > max_iou and current_iou > 0.6:
                max_iou = current_iou
                gt_index_mgt.append({gt_mask_id: mask_id})

    gt_index_gtm = []
    for gt_mask_id, gt_mask in enumerate(gt_masks):
        max_iou = 0
        for mask_id, mask_dict in enumerate(ddup_masks):
            current_iou = iou(gt_mask, mask_dict['segmentation'])
            if current_iou > max_iou and current_iou > 0.5:
                max_iou = current_iou
                gt_index_gtm.append({gt_mask_id: mask_id})
                
    common = [x for x in gt_index_gtm if x in gt_index_mgt]

    # convert gt_index to a dictionary
    gt_dict = {}
    for d in common:
        gt_dict.update(d)
    gt_list = list(gt_dict.keys())

    # relation idx swapping
    relations = data['relations'].copy()
    new_relations = []
    for sublist in relations:
        if all(x in gt_dict for x in sublist[:-1]):
            new_sublist = [gt_dict[x] for x in sublist[:-1]] + [sublist[-1]] 
            new_relations.append(new_sublist)

    ddup_feat = np.array([mask['feat'] for mask in ddup_masks])

    save_entry = {
        'id': img_id,
        'feat': ddup_feat,
        'relations': new_relations,
        'is_train': img_id not in psg_dataset_file['test_image_ids'],
    }
    np.savez(f'./share/feats_0420/save_dict_{img_id}.npz', **save_entry)


def save_random_file(path, id):
    random_filename = f"{id}.txt"
    file_path = os.path.join(path, random_filename)

    with open(file_path, 'w') as f:
        f.write("Process has ended.")

if __name__ == "__main__":
    args = parse_arguments()

    # total_images = len(psg_dataset)
    total_images = args.total_images
    if total_images == 0:
        total_images = len(psg_dataset)
    starting_id = args.start_id
    num_parts = args.num_parts
    images_per_part = total_images // num_parts

    start = args.id * images_per_part + starting_id
    end = start + images_per_part
    img_ids = list(psg_dataset.keys())[start:end]
    
    for img_id in tqdm(img_ids):
        process_image(img_id)
    
    # marking as ended
    save_random_file('.', f'{starting_id}_{args.id}')
