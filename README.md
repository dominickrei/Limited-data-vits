# [WACV 2024] Limited Data, Unlimited Potential: A Study on ViTs Augmented by Masked Autoencoders

---TEASER IMAGE---

## Installation

Create the conda environment and install the necessary packages:

```
conda env create -f environment.yml -n limiteddatavit
```

or alternatively

```
conda create -n limiteddatavit python=3.7 -y
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Data preparation

We provide code for training on ImageNet, CIFAR10, and CIFAR100. CIFAR10 and 100 will be automatically downloaded using torchvision, ImageNet must be downloaded separately. 

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Pretrained model weights
| Model  | Dataset | Evaluation Command|
| ------------- | ------------- | ------------- |
| ViT-T + SSAT ([weights](www.google.com)) | ImageNet-1k | `python main_two_branch.py --data_path /path/to/imagenet/ --resume vitsmall-ssat_imagenet1k_weights.pth --eval --model mae_vit_small` |
| ViT-S + SSAT ([weights](www.google.com))  | ImageNet-1k | `python main_two_branch.py --data_path /path/to/imagenet/ --resume vittiny-ssat_imagenet1k_weights.pth --eval --model mae_vit_tiny` |


## Training models
To train ViT-Tiny with Self-Supervised Auxiliary Task on ImageNet-1k using 8 GPUs run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main_two_branch.py --data_path /path/to/imagenet/ --output_dir ./output_dir --epochs 100 --model mae_vit_tiny
```

Available arguments for `--data_path` are `/path/to/imagenet`, `c10`, `c100`. Other datasets can be added in `utils/datasets.py`.

Available arguments for `--model` are `mae_vit_tiny`, `mae_vit_small`, `mae_vit_base`, `mae_vit_large`, `mae_vit_huge`.

## Citation & Acknowledgement
```
@article{das-limiteddatavit-wacv2024,
    title={Limited Data, Unlimited Potential: A Study on ViTs Augmented by Masked Autoencoders},
    author={Srijan Das and Tanmay Jain and Dominick Reilly and Pranav Balaji and Soumyajit Karmakar and Shyam Marjit and Xiang Li and Abhijit Das and Michael Ryoo},
    journal={2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year={2024}
}
```

This repository is built on top of the code for [Masked Autoencoders Are Scalable Vision Learners:](https://github.com/facebookresearch/mae) from Meta Research.
