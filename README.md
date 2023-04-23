![](./assets/ram_logo.png)

# RAM: Relate-Anything-Model

**[S-lab, NTU](https://www.mmlab-ntu.com/)**

[[`Huggingface Demo`](#method)]

[[`Dataset`](https://psgdataset.org/)]

### 🚀 🚀 🚀 This is a random demo that combine Meta's Segment-Anything model with the ECCV'22 paper: [Panoptic Scene Graph Generation](https://psgdataset.org/). 

### 🔥🔥🔥 Please star our codebase [openpsg](https://github.com/Jingkang50/OpenPSG) and [RAM](https://github.com/Luodian/RelateAnything) if you find it useful / interesting.

Given a random image:
- SAM segments any objects.
- RAM identifies any plausible relations between them.

## Method

To setup the environment, we use conda to manage our dependencies.

You can specify the appropriate cudatoolkit version to install on your machine in the environment.yml file, and then run the following to create the conda environment:

`conda env create -f environment.yml`


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
