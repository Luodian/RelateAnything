![](./assets/ram_logo.png)

# RAM: Relate-Anything-Model
Given a random image:
- SAM segments any objects.
- RAM identifies any plausible relations between them.

## Method

To setup the environment, we use conda to manage our dependencies.

Our developers use CUDA 10.1 to do experiments.

You can specify the appropriate cudatoolkit version to install on your machine in the environment.yml file, and then run the following to create the conda environment:

conda env create -f environment.yml





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
