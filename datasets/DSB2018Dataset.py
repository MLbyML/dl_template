"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
import random

import numpy as np
import tifffile
import torch
from skimage.segmentation import find_boundaries
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset


def convert_to_oneHot(data): # data is 256 x 256
    data_oneHot = onehot_encoding(add_boundary_label(data.astype(np.int32)))
    return data_oneHot


def add_boundary_label(lbl, dtype=np.uint16):
    b = find_boundaries(lbl, mode='outer')
    res = (lbl > 0).astype(dtype)
    res[b] = 2
    return res


def onehot_encoding(lbl, n_classes=3, dtype=np.int32):
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot


class DSB2018Dataset(Dataset):

    def __init__(self, root_dir='./', type="train", bg_id=0, size=None, transform=None):

        print('DSB2018 Dataset created')

        # get image and instance list
        image_list = glob.glob(os.path.join(root_dir, 'dsb-2018/{}/'.format(type), 'images/*.tif'))  # TODO final-1 contains normalized data

        image_list.sort()
        self.image_list = image_list

        instance_list = glob.glob(os.path.join(root_dir, 'dsb-2018/{}/'.format(type), 'masks/*.tif'))  # TODO final-1 contains normalized data

        instance_list.sort()
        self.instance_list = instance_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):

        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image
        image = tifffile.imread(self.image_list[index])
        # sample['image'] = image
        sample['image'] = torch.from_numpy(image[np.newaxis, ...]).float().div(1)  # TODO
        sample['im_name'] = self.image_list[index]

        # load instances
        instance = tifffile.imread(self.instance_list[index])
        # instance, label = self.decode_instance(instance, self.bg_id)
        label_bg, label_fg, label_membrane, instance = self.decode_instance(instance, self.bg_id)

        sample['label_bg'] = torch.from_numpy(label_bg[np.newaxis, ...]).byte()
        sample['label_fg'] = torch.from_numpy(label_fg[np.newaxis, ...]).byte()
        sample['label_membrane'] = torch.from_numpy(label_membrane[np.newaxis, ...]).byte()
        sample['instance'] = torch.from_numpy(instance[np.newaxis, ...]).byte()

        return sample

        # transform
        # if(self.transform is not None):
        #    return self.transform(sample)
        # else:
        #    return sample

    @classmethod
    def decode_instance(cls, pic, bg_id=None):
        pic = np.array(pic, copy=False)

        one_hot_instance_map = convert_to_oneHot(pic)  # last channel is either bg(0), fg(1), membrane(2)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if bg_id is not None:
            mask = pic > bg_id

            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])

                instance_map[mask] = ids
                class_map[mask] = 1

        return one_hot_instance_map[..., 0], one_hot_instance_map[..., 1], one_hot_instance_map[
            ..., 2], instance_map  # check this, is batch dimension included?

