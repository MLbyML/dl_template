import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

import train_config
from datasets import get_dataset
from models import get_model

torch.backends.cudnn.benchmark = True
from utils import AverageMeter, Logger, Visualizer
from scipy import ndimage
from utils2 import matching_dataset

torch.backends.cudnn.benchmark = True

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
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])

val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)


def lambda_(epoch):
    return pow((1 - ((epoch) / args['n_epochs'])), 0.9)


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_, )

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


# create criterion
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 5.0]).cuda())  # apply sigmoid, then softmax
criterion = torch.nn.DataParallel(criterion).to(device)


def train(epoch):
    loss_meter = AverageMeter()
    model.train()
    for i, sample in enumerate(tqdm(train_dataset_it)):
        images = sample['image']  # B 1 Y X
        semantic_masks = sample['semantic_mask']  # B 1 Y X
        semantic_masks.squeeze_(1)  # B Y X (loss expects this format)
        instance = sample['instance_mask']
        output = model(images)
        loss = criterion(output, semantic_masks.long())  # B 1 Y X
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args['display'] and i % args['display_it'] == 0:
            with torch.no_grad():
                visualizer.display(images[0], 'image', 'image')  # 1YX
                output_softmax = F.softmax(output[0], dim=0)
                prediction_fg = output_softmax[1, ...].cpu().detach().numpy()
                pred_fg_thresholded = prediction_fg > 0.5
                instance_map, _ = ndimage.label(pred_fg_thresholded)
                visualizer.display([instance_map, instance[0, 0, ...].cpu()], 'pred', 'predictions', 'groundtruth')
        loss_meter.update(loss.item())

    return loss_meter.avg


def val(epoch):
    loss_meter = AverageMeter()
    average_precision_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):

            images = sample['image']  # B 1 Y X
            semantic_masks = sample['semantic_mask']  # B 1 Y X ==> channel can be 0, 1, 2
            semantic_masks.squeeze_(1)  # B Y X (loss expects this format)
            instance = sample['instance_mask']
            output = model(images)  # B 3 Y X
            loss = criterion(output, semantic_masks.long())  # B 1 Y X
            loss = loss.mean()
            loss_meter.update(loss.item())
            if args['display'] and i % args['display_it'] == 0:
                with torch.no_grad():
                    visualizer.display(images[0], 'image', 'image')  # TODO
                    output_softmax = F.softmax(output[0], dim=0)
                    prediction_fg = output_softmax[1, ...].cpu().detach().numpy()
                    pred_fg_thresholded = prediction_fg > 0.5
                    instance_map, _ = ndimage.label(pred_fg_thresholded)
                    visualizer.display([instance_map, instance[0, 0, ...].cpu()], 'pred', 'predictions', 'groundtruth')
            # compute best iou
            for b in range(output.shape[0]):
                output_softmax = F.softmax(output[0], dim=0)
                prediction_fg = output_softmax[1, ...].cpu().detach().numpy()
                pred_fg_thresholded = prediction_fg > 0.5
                instance_map, _ = ndimage.label(pred_fg_thresholded)
                sc = matching_dataset([instance_map], [instance[b, 0, ...].cpu().detach().numpy()], thresh=0.5,
                                      show_progress=False)
                average_precision_meter.update(sc.accuracy)

    return loss_meter.avg, average_precision_meter.avg


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
