# Controlling the Fidelity and Diversity of Deep Generative Models via Pseudo Density

---

[![TMLR](https://img.shields.io/badge/TMLR-2024-green)](https://www.jmlr.org/tmlr/)
[![Poster](https://img.shields.io/badge/ICLR%202025-Poster-blue)](https://iclr.cc/Conferences/2025) 
[![arXiv](https://img.shields.io/badge/arXiv-2411.18810-b31b1b.svg)](https://arxiv.org/abs/2411.18810) 

[Shuangqi Li](mailto:shuangqi.li@epfl.ch)¹ · [Chen Liu](mailto:chen.liu@cityu.edu.hk)² · [Tong Zhang](mailto:tong.zhang@epfl.ch)¹ · [Hieu Le](mailto:minh.le@epfl.ch)¹ · [Sabine Süsstrunk](mailto:sabine.susstrunk@epfl.ch)¹ · [Mathieu Salzmann](mailto:mathieu.salzmann@epfl.ch)¹

¹School of Computer and Communication Sciences, EPFL, Switzerland
²Department of Computer Science, City University of Hong Kong, Hong Kong

## Overview

This repository contains the official implementation of our TMLR paper "Controlling the Fidelity and Diversity of Deep Generative Models via Pseudo Density", presented at ICLR 2025.

### Abstract

We introduce an approach to bias deep generative models, such as GANs and diffusion models, towards generating data with either enhanced fidelity or increased diversity.
Our approach involves manipulating the distribution of training and generated data through a novel metric for individual samples, named pseudo density, which is based on the nearest-neighbor information from real samples.
Our approach offers three distinct techniques to adjust the fidelity and diversity of deep generative models:
1) Per-sample perturbation, enabling precise adjustments for individual samples towards either more common or more unique characteristics;
2) Importance sampling during model inference to enhance either fidelity or diversity in the generated data;
3) Fine-tuning with importance sampling, which guides the generative model to learn an adjusted distribution, thus controlling fidelity and diversity.

Furthermore, our fine-tuning method demonstrates the ability to improve the Frechet Inception Distance (FID) for pre-trained generative models with minimal iterations.

## Requirements

For extracting the image features, we used the `timm` library of the version `0.5.4`. Please note that later versions might have some significant changes, leading to different results.

In addition, this repository, except `get_density_and_estimator.ipynb`, is built upon **StyleGAN2**. If you want to reproduce our results in the paper, please refer to the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) codebase for the dependencies and resources.

## Usage

### Compute pseudo density

To compute the pseudo density, as well as the density thresholds, of a set of images, you can run the first section in `get_density_and_estimator.ipynb`.

### Train a density estimator

To train a density estimator, you can run the second section in `get_density_and_estimator.ipynb`.

### Density-based perturbation

To perform the density-based perturbation, you can run the third section in `get_density_and_estimator.ipynb`.

### Importance sampling during inference StyleGAN2

```bash
python calc_metrics.py --metrics=pr50k3_full_quality \
                       --network=YOUR_MODEL_PICKLE_FILE \
                       --data=YOUR_FFHQ1024_DATASET_PATH \
                       --gpus=1
```

!!! For now, many hyperparameters and file paths are hard-coded in the `metrics/metric_main.py`, `metrics/metric_utils.py`, and `metrics/precision_recall.py` files. We will improve this in the future if this repository interests more people.

### Fine-tuning StyleGAN2 with importance sampling

```bash
# To improve fidelity. 
# (The configs corresponding to the top-right of Table 1 in the paper)
python train.py --outdir=YOUR_OUTPUT_DIR_FOR_FINETUNED_MODEL \
                --data=YOUR_FFHQ1024_DATASET_PATH \
                --gpus=2 \
                --aug=noaug \
                --mirror=1 \
                --resume=YOUR_PRETRAINED_MODEL_PICKLE_FILE \
                --cfg=stylegan2 \
                --metrics=pr50k3_full,fid50k_full \
                --snap=10 \
                --sample-threshold-real=1.1869 \
                --sample-weight-real=33.0 \
                --sample-threshold-fake=1.1869 \
                --sample-weight-fake=0.33

# To improve FID. 
# (The configs corresponding to the top-right of Table 1 in the paper)
python train.py --outdir=YOUR_OUTPUT_DIR_FOR_FINETUNED_MODEL \
                --data=YOUR_FFHQ1024_DATASET_PATH \
                --gpus=2 \
                --aug=noaug \
                --mirror=1 \
                --resume=YOUR_PRETRAINED_MODEL_PICKLE_FILE \
                --cfg=stylegan2 \
                --metrics=pr50k3_full,fid50k_full \
                --snap=1 \
                --kimg=12 \
                --sample-threshold-real=0.9444 \
                --sample-weight-real=0.5 \
                --sample-threshold-fake=0.9444 \
                --sample-weight-fake=1

```

!!! For now, many hyperparameters and file paths are hard-coded in the `training/attacks.py` file. We will improve this in the future if this repository interests more people.

Nevertheless, our method is very simple to implement in your own training/inference pipelines, once you obtained the pseudo density for your own dataset and trained a density estimator (which only takes a few minutes).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{li2024controlling,
  title={Controlling the Fidelity and Diversity of Deep Generative Models via Pseudo Density},
  author={Li, Shuangqi and Liu, Chen and Zhang, Tong and Le, Hieu and S{\"u}sstrunk, Sabine and Salzmann, Mathieu},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```



## Acknowledgements

We created our code based on the following repositories:
- [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
