import os

import torch
from scipy import ndimage
from tifffile import imsave
from tqdm import tqdm

import test_config
from datasets import get_dataset
from models import get_model
from utils2 import matching_dataset

torch.backends.cudnn.benchmark = True
import numpy as np
import torch.nn.functional as F

args = test_config.get_args()

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(args['dataset']['name'], args['dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# load snapshot
if os.path.exists(args['checkpoint_path']):
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
else:
    assert (False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

# check if static exists
if not os.path.isdir(args['save_dir']):
    os.mkdir(args['save_dir'])
    os.mkdir(args['save_dir']+'/instances')
    os.mkdir(args['save_dir']+'/gt')
    os.mkdir(args['save_dir'] + '/results')

model.eval()

optimal_thresh = args['optimal_thresh']
ap_thresh = args['ap_thresh']
min_object_size = args['min_object_size']

with torch.no_grad():
    result_list = []
    image_file_names = []
    for sample in tqdm(dataset_it):

        im = sample['image']
        h = im.shape[2]
        w = im.shape[3]
        im = im[..., :4 * (h // 4), :4 * (w // 4)]
        label = sample['semantic_mask'][..., :4 * (h // 4), :4 * (w // 4)]  # B 1 Y X
        instance = sample['instance_mask'][..., :4 * (h // 4), :4 * (w // 4)]  # B 1 Y X
        output = model(im)  # B3YX

        output_softmax = F.softmax(output[0], dim=0)
        prediction_fg = output_softmax[1, ...].cpu().detach().numpy()
        pred_fg_thresholded = prediction_fg > 0.5
        instance_map, _ = ndimage.label(pred_fg_thresholded)

        instance_map, nb = ndimage.label(pred_fg_thresholded)
        instance_map_filtered = np.zeros_like(instance_map)

        for item in np.unique(instance_map)[1:]:
            if ((instance_map == item).sum() < min_object_size):
                instance_map_filtered[instance_map == item] = 0
            else:
                instance_map_filtered[instance_map == item] = item

        results = matching_dataset([instance_map_filtered], [instance[0, 0, ...].cpu().detach().numpy()],
                                   thresh=ap_thresh)
        print("AP @ {} = {:.3f}".format(str(ap_thresh), results.accuracy))
        result_list.append(results.accuracy)  # TODO
        base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
        image_file_names.append(base)
        # do for each image

        if args['save_images'] and ap_thresh == 0.5:
            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            instances_file = os.path.join(args['save_dir'], 'instances/', base + '.tif')  # TODO
            imsave(instances_file, instance_map_filtered)
            gt_file = os.path.join(args['save_dir'], 'gt/', base + '.tif')  # TODO
            imsave(gt_file, instance[0, 0, ...].cpu().detach().numpy())

    # do for the complete set of images
    if args['save_results']:
        txt_file = os.path.join(args['save_dir'], 'results/combined_AP' + str(ap_thresh) + '.txt')
        with open(txt_file, 'w') as f:
            f.writelines("ImageFileName, Optimal Threshold, MinObectSize, Intersection Threshold, accuracy \n")
            f.writelines("+++++++++++++++++++++++++++++++++\n")
            for ind, im_name in enumerate(image_file_names):
                im_name_png = im_name + '.png'
                score = result_list[ind]
                f.writelines("{} {:.02f} {:.02f} {:.02f} {:.02f}\n".format(im_name_png, optimal_thresh, min_object_size,
                                                                           ap_thresh, score))
            f.writelines("+++++++++++++++++++++++++++++++++\n")
            f.writelines(
                "Average Precision (AP) {:.02f} {:.02f} {:.02f} {:.03f}\n".format(optimal_thresh, min_object_size,
                                                                                  ap_thresh, np.mean(result_list)))
print("mean result", np.mean(result_list))
