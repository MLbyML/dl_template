"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

import test_config
import torch
from datasets import get_dataset
from models import get_model
from utils2 import matching_dataset
from scipy import ndimage
torch.backends.cudnn.benchmark = True
import numpy as np  # TODO
import tifffile

args = test_config.get_args()

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# if args['save']:
#     if not os.path.exists(args['save_dir']):
#         os.makedirs(args['save_dir'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs'])

dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# load snapshot
if os.path.exists(args['checkpoint_path']):
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
else:
    assert (False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

model.eval()


with torch.no_grad():
    threshold = np.arange(0.2, 0.7, 0.05)
    minObjectSize=np.arange(30, 200, 10)
    resultArray=np.zeros((len(dataset), len(threshold), len(minObjectSize)))
    for indk, k in enumerate(threshold):
        for indj, j in enumerate(minObjectSize):
            print("threshold is: ", k, "minObjectSize is", j)
            for indsample, sample in enumerate(dataset_it):
                im = sample['image']  # B 1 Y X
                label_bg = sample['label_bg']  # B 1 YX
                label_fg = sample['label_fg']  # B 1 YX
                label_membrane = sample['label_membrane']  # B 1 YX
                instance = sample['instance']  # B 1  Y X
                output = model(im)  # B3YX
                pred_numpy = output[0].cpu().detach().numpy()  # 3YX
                pred_numpy = np.moveaxis(pred_numpy, 0, -1)  # YX3
                prediction_exp = np.exp(pred_numpy[..., :])
                prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
                prediction_bg = prediction_seg[..., 0]
                prediction_fg = prediction_seg[..., 1]
                prediction_membrane = prediction_seg[..., 2]
                pred_thresholded = prediction_fg > k  # TODO optimize over val
                instance_map, nb = ndimage.label(pred_thresholded)
                instance_map_filtered=np.zeros_like(instance_map)
                for item in np.unique(instance_map)[1:]:
                    if ((instance_map == item).sum() < j):
                        instance_map_filtered[instance_map == item] = 0
                    else:
                        instance_map_filtered[instance_map == item] = item

                sc = matching_dataset([instance_map_filtered], [instance[0, 0, ...].cpu().detach().numpy()], thresh=0.5)
                resultArray[indsample, indk, indj]=sc.accuracy


    print("result array shape", resultArray.shape)
    meanResult=np.mean(resultArray, axis=0)
    print(meanResult)
    bestIndThreshold, bestIndObjectSize = np.unravel_index(np.argmax(meanResult), meanResult.shape)
    print("Best accuracy is ", np.max(meanResult))
    print("Best metaparams are", threshold[bestIndThreshold], minObjectSize[bestIndObjectSize])