# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch
import math
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop
import torchvision



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultiCropsROITransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, small_global_crop, q_transform, small_transform, k_transform):
        self.small_global_crop = small_global_crop
        self.q_transform = q_transform
        self.k_transform = k_transform
        self.small_transform = small_transform

    def __call__(self, x):
        img = x
        global_imgs, small_imgs, small_bbox = self.small_global_crop(img)

        q = self.q_transform(global_imgs[0])
        k = self.k_transform(global_imgs[1])
        small_q = [self.small_transform(small_img) for small_img in small_imgs]

        return [q, k] + small_q + small_bbox


class SmallGlobalRandomResizedCrop(RandomResizedCrop):
    def __init__(self, flip_p=0.5, num_small=5, small_size=64, small_scale=(0.05, 0.6), small_ratio=(0.75, 1.3333333333333), **kwargs):
        super().__init__(**kwargs)
        self.num_small = num_small
        # HACK: the size must be the tuple, int only resize height or width of the image
        self.small_size = (small_size, small_size)
        self.small_scale = small_scale
        # assert self.small_scale[1] <= self.scale[0]
        self.small_ratio = small_ratio
        self.flip_p = flip_p

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        if torchvision.__version__ == "0.12.0":
            width, height = F.get_image_size(img)
        else:
            width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            global_scale = torch.empty(1).uniform_(scale[0], scale[1]).item()
            target_area = area * global_scale
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        global_imgs = []
        for index in range(2):
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            global_img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
            if index == 0:
                if torch.rand(1) < self.flip_p:
                    global_img = F.hflip(global_img)
            global_imgs.append(global_img)

        part_img = []
        small_bbox = []
        for _ in range(self.num_small):
            small_i, small_j, small_h, small_w = self.get_params(global_imgs[0], self.small_scale, self.small_ratio)
            small_bbox.append(torch.Tensor([small_j, small_i, small_j + small_w, small_i + small_h]))
            part_img.append(F.resized_crop(global_imgs[0], small_i, small_j, small_h, small_w, self.small_size, self.interpolation))

        return global_imgs, part_img, small_bbox
