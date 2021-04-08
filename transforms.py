import collections
import numpy as np
import torch
from torchvision.transforms import transforms as T

class RandomRotationsAndFlips(T.RandomRotation):
    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
    def __call__(self, sample):

        angle = self.get_params(self.degrees)
        times = np.random.choice(4)
        flip = np.random.choice(2)

        for idx, k in enumerate(self.keys):
            assert (k in sample)
            temp = np.ascontiguousarray(np.rot90(sample[k], times, (1, 2)))
            if flip == 0:
                sample[k] = temp
            else:
                sample[k] = np.ascontiguousarray(np.flip(temp, axis=1))  # flip about Y - axis
        return sample


class ToTensorFromNumpy(object):
    def __init__(self, keys=[], type="float", normalization_factor=255):

        if isinstance(type, collections.Iterable):
            assert (len(keys) == len(type))

        self.keys = keys
        self.type = type
        self.normalization_factor = normalization_factor

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):
            assert (k in sample)

            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]
            if t == torch.FloatTensor:  # image
                sample[k] = torch.from_numpy(sample[k].astype("float32")).float().div(self.normalization_factor)  # images
            elif t == torch.ByteTensor or t ==torch.ShortTensor:  # instance, label
                sample[k] = torch.from_numpy(sample[k]).short()
        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']

        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)
