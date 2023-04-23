<p align="center" width="100%">
<img src="https://i.postimg.cc/25yJ4qFp/ram-logo.png"  width="60%" height="30%">
</p>

# RAM: Relate-Anything-Model

**Developers (alphabetical order):** 
[Zujin Guo](https://scholar.google.com/citations?user=G8DPsoUAAAAJ&hl=zh-CN), 
[Bo Li](https://brianboli.com/), 
[Jingkang Yang](https://jingkang50.github.io/), 
[Zijian Zhou](https://sites.google.com/view/zijian-zhou/home).

**Affiliate: [MMLab@NTU](https://www.mmlab-ntu.com/)** & **[VisCom Lab, KCL](https://viscom.nms.kcl.ac.uk/)**

---

🚀 🚀 🚀 This is a random demo that combine Meta's Segment-Anything model with the ECCV'22 paper: [Panoptic Scene Graph Generation](https://psgdataset.org/). 

🔥🔥🔥 Please star our codebase [OpenPSG](https://github.com/Jingkang50/OpenPSG) and [RAM](https://github.com/Luodian/RelateAnything) if you find it useful/interesting.

[[`Huggingface Demo`](#method)]

[[`Dataset`](https://psgdataset.org/)]

Relate Anything Model is capable of taking an image as input and utilizing SAM to identify the corresponding mask within the image. Subsequently, RAM can provide an analysis of the relationship between any arbitrary objects mask.

[![demo.png](https://i.postimg.cc/CKh8tSB4/demo.png)](https://postimg.cc/k2HDRryV)

## Examples

Our current demo supports:

(1) generate arbitary objects masks and reason relationships in between. 

(2) given coordinates then generate object masks and reason the relationship between given objects and other objects in the image.

We will soon add support for detecting semantic labels of objects with the help of [OVSeg](https://github.com/facebookresearch/ov-seg).

Here are some examples of the Relate Anything Model in action about playing soccer, dancing, and playing basketball.

<!-- ![](./assets/basketball.gif) -->

![](./assets/basketball.png)

![](./assets/soccer.png)

![](https://i.postimg.cc/43VkhRNp/shaking-hands.png)

[![collie.png](https://i.postimg.cc/zvV1vbLG/collie.png)](https://postimg.cc/hQWY3Gbk)

![](https://i.postimg.cc/9QpRyK8w/coord.png)

## Method

RAM utilizes the Segment Anything Model (SAM) to accurately mask objects within an image, and subsequently extract features corresponding to the segmented regions. Employ a Transformer module to facilitate feature interaction among distinct objects, and ultimately compute pairwise object relationships, thereby categorizing their interrelations.

## Setup

To set up the environment, we use Conda to manage dependencies.
To specify the appropriate version of cudatoolkit to install on your machine, you can modify the environment.yml file, and then create the Conda environment by running the following command:

```bash
conda env create -f environment.yml
```

Make sure to use `segment_anything` in this repository, which includes the mask feature extraction operation.

Download the pretrained model
1. SAM: [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
2. RAM: [link](https://1drv.ms/u/s!AgCc-d5Aw1cumQapZwcaKob8InQm?e=qyMeTS)

Place these two models in `./checkpoints/` from the root directory.

Run our demo locally by running the following command:

```bash
python app.py
```

## Developers

We have equally contributed to this project in our spare time, in alphabetical order.
[Zujin Guo](https://scholar.google.com/citations?user=G8DPsoUAAAAJ&hl=zh-CN), 
[Bo Li](https://brianboli.com/), 
[Jingkang Yang](https://jingkang50.github.io/), 
[Zijian Zhou](https://sites.google.com/view/zijian-zhou/home).

**[MMLab@NTU](https://www.mmlab-ntu.com/)** & **[VisCom Lab, KCL](https://viscom.nms.kcl.ac.uk/)**

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@inproceedings{yang2022psg,
    author = {Yang, Jingkang and Ang, Yi Zhe and Guo, Zujin and Zhou, Kaiyang and Zhang, Wayne and Liu, Ziwei},
    title = {Panoptic Scene Graph Generation},
    booktitle = {ECCV}
    year = {2022}
}

@inproceedings{yang2023pvsg,
    author = {Yang, Jingkang and Peng, Wenxuan and Li, Xiangtai and Guo, Zujin and Chen, Liangyu and Li, Bo and Ma, Zheng and Zhou, Kaiyang and Zhang, Wayne and Loy, Chen Change and Liu, Ziwei},
    title = {Panoptic Video Scene Graph Generation},
    booktitle = {CVPR},
    year = {2023},
}
```
