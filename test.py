import os
from matplotlib import pyplot as plt
from tqdm import tqdm

import test_config
import torch
from datasets import get_dataset
from models import get_model
from utils2 import matching_dataset
from scipy import ndimage
from tifffile import imsave
torch.backends.cudnn.benchmark = True
import numpy as np  # TODO

args = test_config.get_args()

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

optimal_thresh=args['optimal_thresh']
ap_thresh=args['ap_thresh']
minObjectSize=args['minObjectSize']

with torch.no_grad():
    resultList = []  # TODO
    imageFileNames=[]
    for sample in tqdm(dataset_it):

        im = sample['image']
        h=im.shape[2]
        w=im.shape[3]
        im=im[..., :4*(h//4), :4*(w//4)]
        # B 1 Y X #TODO
        label_bg = sample['label_bg'][..., :4*(h//4), :4*(w//4)]  # B 1 YX
        label_fg = sample['label_fg'][..., :4*(h//4), :4*(w//4)]  # B 1 YX
        label_membrane = sample['label_membrane'][..., :4*(h//4), :4*(w//4)]  # B 1 YX
        instance = sample['instance'][..., :4*(h//4), :4*(w//4)] # B 1  Y X
        output = model(im)  # B3YX

        pred_numpy = output[0].cpu().detach().numpy() # 3YX
        pred_numpy = np.moveaxis(pred_numpy, 0, -1)   # YX3
        prediction_exp = np.exp(pred_numpy[..., :])
        prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
        prediction_bg = prediction_seg[..., 0]
        prediction_fg = prediction_seg[..., 1]
        prediction_membrane = prediction_seg[..., 2]
        pred_thresholded = prediction_fg > optimal_thresh # TODO optimize over val

        instance_map, nb = ndimage.label(pred_thresholded)
        ## new addition
        ## ignore items below a certain minSize
        instance_map_filtered=np.zeros_like(instance_map)

        for item in np.unique(instance_map)[1:]:
            if((instance_map==item).sum()<minObjectSize):
                instance_map_filtered[instance_map==item]=0
            else:
                instance_map_filtered[instance_map==item]=item



        sc = matching_dataset([instance_map_filtered], [instance[0, 0, ...].cpu().detach().numpy()], thresh=ap_thresh)  # TODO
        print("sc", sc.accuracy, flush=True)
        resultList.append(sc.accuracy)  # TODO
        base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
        imageFileNames.append(base)
        # do for each image

        if args['saveImages'] and ap_thresh==0.5:
            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            instances_file = os.path.join(args['save_dir'], 'instances/', base + '.tif')  # TODO
            imsave(instances_file, instance_map_filtered)
            gt_file =os.path.join(args['save_dir'], 'gt/', base + '.tif')  # TODO
            imsave(gt_file, instance[0, 0, ...].cpu().detach().numpy())

    # do for the complete set of images
    if args['saveResults']:
        txt_file = os.path.join(args['save_dir'], 'results/combined_AP' + str(ap_thresh)+'.txt')
        with open(txt_file, 'w') as f:
            f.writelines("ImageFileName, Optimal Threshold, MinObectSize, Intersection Threshold, accuracy \n")
            f.writelines("+++++++++++++++++++++++++++++++++\n")
            for ind, im_name in enumerate(imageFileNames):
                im_name_png = im_name +'.png'
                score = resultList[ind]
                f.writelines("{} {:.02f} {:.02f} {:.02f} {:.02f}\n".format(im_name_png, optimal_thresh, minObjectSize, ap_thresh, score))
            f.writelines("+++++++++++++++++++++++++++++++++\n")
            f.writelines("Average Precision (AP) {:.02f} {:.02f} {:.02f} {:.03f}\n".format(optimal_thresh, minObjectSize, ap_thresh, np.mean(resultList)))
print("mean result", np.mean(resultList))
