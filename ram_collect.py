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

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='ID of the part to process')
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
    gt_feats = []
    for gt_mask in gt_masks:
        max_iou = 0
        best_feat = None

        for mask_dict in ddup_masks:
            current_iou = iou(gt_mask, mask_dict['segmentation'])

            if current_iou > max_iou:
                max_iou = current_iou
                best_feat = mask_dict['feat']

        gt_feats.append(best_feat)

    gt_feats = np.array(gt_feats)
    
    save_entry = {
        'id': img_id,
        'feat': gt_feats,
        'relations': data['relations'],
        'is_train': img_id in psg_dataset_file['test_image_ids'],
    }
    np.savez(f'./feats/save_dict_{img_id}.npz', **save_entry)


if __name__ == "__main__":
    args = parse_arguments()

    total_images = len(psg_dataset)
    num_parts = 25
    images_per_part = total_images // num_parts

    start = args.id * images_per_part
    end = start + images_per_part if args.id < num_parts - 1 else None

    img_ids = list(psg_dataset.keys())[start:end]

    for img_id in tqdm(img_ids):
        process_image(img_id)
