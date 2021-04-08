"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os
import torch


DATA_DIR = os.environ.get('DATA_DIR')  # TODO

args = dict(

    cuda=True,
    display=False,
    optimal_thresh = 0.5,
    minObjectSize=0,
    ap_thresh=0.5,
    saveResults=True,  # TODO\
    saveImages=True,
    save_dir='./../static/',
    checkpoint_path='./exp/dsb2018_october3/checkpoint.pth',  # TODO
    dataset={
        'name': 'dsb2018',  # TODO
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'test',

        }
    },

    model={
        'name': 'unet',
        'kwargs': {
            'num_classes': 3,
            'depth': 3,
        }
    }
)


def get_args():
    return copy.deepcopy(args)
