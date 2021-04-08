import copy
import os
import torch
import transforms as my_transforms
DATA_DIR = os.environ.get('DATA_DIR')  # TODO: specify correctly --> for Unet, do normalized by min-max!
args = dict(

    cuda=True,
    display=False,  # TODO
    display_it=5,
    save=True,
    save_dir='./exp/dsb2018_april8_v0.1.1', #TODO : specify
    resume_path=None,  # TODO

    train_dataset={
        'name': 'dsb2018',  # TODO
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'train',
            'size': 3000,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomRotationsAndFlips',
                    'opts': {
                        'keys': ('image', 'instance_mask', 'semantic_mask'),
                        'degrees': 90,

                    }
                },
                {
                    'name': 'ToTensorFromNumpy',
                    'opts': {
                        'keys': ('image', 'instance_mask', 'semantic_mask'),
                        'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor),
                        'normalization_factor': 1  # TODO --> for unet, use normalized by min-max
                    }
                },
            ])
        },
        'batch_size': 16,
        'workers': 8  # TODO 8
    },

    val_dataset={
        'name': 'dsb2018',  # TODO
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'val',
            'size': 2900,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomRotationsAndFlips',
                    'opts': {
                        'keys': ('image', 'instance_mask', 'semantic_mask'),
                        'degrees': 90,
                    }
                },
                {
                    'name': 'ToTensorFromNumpy',
                    'opts': {
                        'keys': ('image', 'instance_mask', 'semantic_mask'),
                        'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor),
                        'normalization_factor': 1 # TODO --> for unet, use normalized by min-max
                    }
                },
            ])
        },
        'batch_size': 16,
        'workers': 8
    },

    model={
        'name': 'unet',
        'kwargs': {
            'num_classes': 3,
            'depth': 3,
        }
    },

    lr=5e-4,
    n_epochs=200,  # TODO
)


def get_args():
    return copy.deepcopy(args)
