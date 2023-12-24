# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
import torch

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_path == 'c10':
        dataset = datasets.CIFAR10(
                    root='./data', train=is_train, download=True, transform=transform
        )
    elif args.data_path == 'c100':
        dataset = datasets.CIFAR100(
                    root='./data', train=is_train, download=True, transform=transform
        )
    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

    if is_train and args.subset_size != 1:
        np.random.seed(127)

        dsize = len(dataset)
        idxs = np.random.choice(dsize, int(dsize*args.subset_size), replace=False)

        dataset = torch.utils.data.Subset(dataset, idxs)
        print('Subset dataset size: ', len(dataset))

    print(f'{dataset} ({len(dataset)})')

    return dataset


def build_transform(is_train, args):
    if args.data_path == 'c10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    if 'perturb_perspective' in vars(args) and args.perturb_perspective:
        print('[Log] Perturbing perspective of images in dataset')
        t.append(transforms.RandomPerspective(distortion_scale=0.5, p=1.0))
    else:
        print('[Log] Not perturbing perspective of images in dataset')

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
