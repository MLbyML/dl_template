import copy
import os
import torch
import transforms as my_transforms
DATA_DIR = os.environ.get('DATA_DIR')  # TODO

args = dict(

    cuda=True,
    display=False,
    optimal_thresh=0.5,
    min_object_size=21, #TODO--> fix by looking at train and val
    ap_thresh=0.5,
    save_results=True,  # TODO\
    save_images=True,
    save_dir='./static/',
    checkpoint_path='./exp/dsb2018_april8_v0.1.1/best_iou_model.pth',  # TODO
    dataset={
        'name': 'dsb2018',  # TODO
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'test',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensorFromNumpy',
                    'opts': {
                        'keys': ('image', 'instance_mask', 'semantic_mask'),
                        'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor),
                        'normalization_factor': 1
                    }
                },
            ]),

        },

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
