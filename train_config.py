"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

import torch
# from utils import transforms as my_transforms

DATA_DIR=os.environ.get('DATA_DIR') # TODO


args = dict(

    cuda=True,
    display=False, #TODO
    display_it=5,
    save=True,
    save_dir='./exp/dsb2018_april8_v0.1.1',
    resume_path=None, # None

    train_dataset = {
        'name': 'dsb2018',#TODO
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'train',
            'size': 3000,
            # 'transform': my_transforms.get_transform([
            #     {
            #         'name': 'ToTensor',
            #         'opts': {
            #             'keys': ('image', 'instance', 'label'),
            #             'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
            #         }
            #     },
            # ]),
        },
        'batch_size': 16,
        'workers': 8 #TODO 8
    },

    val_dataset = {
        'name':'dsb2018', #TODO
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'val',
            'size': 2900,
            # 'transform': my_transforms.get_transform([
            #     {
            #         'name': 'ToTensor',
            #         'opts': {
            #             'keys': ('image', 'instance', 'label'),
            #             'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
            #         }
            #     },
            # ]),
        },
        'batch_size': 16,
        'workers': 8
    },

    model = {
        'name': 'unet',
        'kwargs': {
            'num_classes': 3,
            'depth': 3,
        }
    },

    lr=5e-4,
    n_epochs=200, #TODO
    loss_opts={
        # nothing
    },
    loss_w={
        #nothing
    },
)


def get_args():
    return copy.deepcopy(args)
