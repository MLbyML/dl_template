"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import train_config
from datasets import get_dataset
from models import get_model
import torch.nn as nn
torch.backends.cudnn.benchmark = True
from utils import AverageMeter, Logger, Visualizer
from scipy import ndimage
import numpy as np
from utils2 import matching_dataset

args = train_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])


train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)


# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])

val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)


# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)

def lambda_(epoch):
    return pow((1-((epoch)/args['n_epochs'])), 0.9)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_,)


visualizer = Visualizer(('image', 'pred'))

# Logger
logger = Logger(('train', 'val', 'val ap'), 'loss')


# resume
start_epoch = 0
best_ap = 0
if args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    start_epoch = state['epoch'] + 1
    best_ap = state['best_ap']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    logger.data = state['logger_data']


def save_checkpoint(state, is_best, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth'))

def lossFunctionSegmentation(prediction, labels_BG, labels_FG, labels_M):

    onehot_targets = torch.cat((labels_BG, labels_FG, labels_M), 1).cuda()
    class_weights = torch.tensor([1.0, 1.0, 5.0]).cuda()
    criterion1 = nn.CrossEntropyLoss(weight=class_weights)
    multiclass_targets = torch.argmax(onehot_targets, dim=1)
    loss = criterion1(prediction, multiclass_targets.long())
    return loss



def train(epoch):
    loss_meter = AverageMeter()
    model.train()
    for i, sample in enumerate(tqdm(train_dataset_it)):
        im = sample['image'] # B 1 Y X
        label_bg = sample['label_bg']# B 1 Y X
        label_fg = sample['label_fg'] # B 1 Y X
        label_membrane = sample['label_membrane'] # B 1 Y X
        instance = sample['instance'] # B 1 Y X
        output = model(im) # B3YX
        loss= lossFunctionSegmentation(output, label_bg, label_fg, label_membrane)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args['display'] and i % args['display_it'] == 0:
            with torch.no_grad():
                visualizer.display(im[0], 'image', 'image')  # TODO

                pred_numpy = output[0].cpu().detach().numpy()  # 3YX
                pred_numpy = np.moveaxis(pred_numpy, 0, -1)  # YX3
                prediction_exp = np.exp(pred_numpy[..., :])
                prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
                prediction_fg = prediction_seg[..., 1]
                pred_thresholded = prediction_fg > 0.5  # TODO optimize over val
                instance_map, _ = ndimage.label(pred_thresholded)
                visualizer.display([instance_map, instance[0, 0, ...].cpu()], 'pred', 'predictions', 'groundtruth')  # TODO
        loss_meter.update(loss.item())




    return loss_meter.avg

def val(epoch):
    loss_meter, ap_meter = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample['image']
            label_bg = sample['label_bg']  # BYX
            label_fg = sample['label_fg']  # BYX
            label_membrane = sample['label_membrane']  # BYX
            instance = sample['instance']
            output = model(im)  # B3YX
            loss = lossFunctionSegmentation(output, label_bg, label_fg, label_membrane)
            loss = loss.mean()
            if args['display'] and i % args['display_it'] == 0:
                with torch.no_grad():
                    visualizer.display(im[0], 'image', 'image')  # TODO

                    pred_numpy = output[0].cpu().detach().numpy()  # 3YX
                    pred_numpy = np.moveaxis(pred_numpy, 0, -1)  # YX3
                    prediction_exp = np.exp(pred_numpy[..., :])
                    prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
                    prediction_fg = prediction_seg[..., 1]
                    pred_thresholded = prediction_fg > 0.5  # TODO optimize over val
                    instance_map, _ = ndimage.label(pred_thresholded)
                    visualizer.display([instance_map, instance[0, 0, ...].cpu()], 'pred', 'predictions',  'groundtruth')  # TODO

            # compute best iou
            for b in range(output.shape[0]):
                pred_numpy = output[b].cpu().detach().numpy()  # 3YX not entirely accurate -  we are ignoring all elements of the batch!
                pred_numpy = np.moveaxis(pred_numpy, 0, -1)  # YX3
                prediction_exp = np.exp(pred_numpy[..., :])
                prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
                prediction_fg = prediction_seg[..., 1]
                pred_thresholded = prediction_fg > 0.5  # TODO optimize over val
                instance_map, _ = ndimage.label(pred_thresholded)
                sc = matching_dataset([instance_map], [instance[b, 0, ...].cpu().detach().numpy()], thresh=0.5, show_progress=False)  # TODO
                loss_meter.update(loss.item())
                ap_meter.update(sc.accuracy)

    return loss_meter.avg,  ap_meter.avg




for epoch in range(start_epoch, args['n_epochs']):

    print('Starting epoch {}'.format(epoch))
    scheduler.step(epoch)

    train_loss = train(epoch)
    val_loss, val_ap = val(epoch)

    print('===> train loss: {:.2f}'.format(train_loss))
    print('===> val loss: {:.2f}, val ap: {:.2f}'.format(val_loss, val_ap))
    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.add('val ap', val_ap)
    logger.plot(save=args['save'], save_dir=args['save_dir'])


    is_best = val_ap > best_ap
    best_ap = max(val_ap, best_ap)

    if args['save']:
        state = {
            'epoch': epoch,
            'best_ap': best_ap,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data,

        }
        save_checkpoint(state, is_best)
