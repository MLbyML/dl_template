import numpy as np
import tifffile
import torch
import torch.nn as nn
from glob import glob
from skimage.segmentation import find_boundaries, relabel_sequential
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import shutil, os
from scipy import ndimage
from utils import matching_dataset
from UNet_original import UNet
#from UNet import UNet
torch.backends.cudnn.benchmark = True

def train():
    model.train()
    for i, sample in enumerate(tqdm(train_dataset_iterator)):
        loss_meter = []
        images = sample['image']  # B 1 Y X
        semantic_masks = sample['semantic_mask']  # B 1 Y X
        semantic_masks.squeeze_(1) # B Y X (loss expects this format)
        output = model(images)
        loss = criterion(output, semantic_masks.long())  # B 1 Y X
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.append(loss.item())
    return np.mean(loss_meter)

    
def val():
    loss_meter = []
    average_precision_meter = []
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_iterator)):
            loss_meter = []
            images = sample['image']  # B 1 Y X
            semantic_masks = sample['semantic_mask']  # B 1 Y X ==> channel can be 0, 1, 2
            semantic_masks.squeeze_(1) # B Y X (loss expects this format)
            instance = sample['instance_mask']
            output = model(images)
            loss = criterion(output, semantic_masks.long())  # B 1 Y X
            loss = loss.mean()
            loss_meter.append(loss.item())
            # compute best iou
            for b in range(output.shape[0]):
                pred_numpy = output[b].cpu().detach().numpy()  
                prediction_exp = np.exp(pred_numpy[:, ...])
                prediction_seg = prediction_exp / np.sum(prediction_exp, axis=0)[np.newaxis, ...]
                prediction_fg = prediction_seg[1, ...]
                pred_thresholded = prediction_fg > 0.5  
                instance_map, _ = ndimage.label(pred_thresholded)
                sc = matching_dataset([instance_map], [instance[b, 0, ...].cpu().detach().numpy()], thresh=0.5, show_progress=False)  
                average_precision_meter.append(sc.accuracy)

    return np.mean(loss_meter), np.mean(average_precision_meter)

def convert_to_class_labels(lbl):
    b = find_boundaries(lbl, mode='outer') # outer and inner make a lot of difference!
    res = (lbl > 0).astype(np.uint8)
    res[b] = 2
    return res



class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=None, transform = None):
        self.image_name_list = sorted(glob(image_dir + '*.tif'))
        self.mask_name_list = sorted(glob(mask_dir + '*.tif'))
        self.size = size
        self.real_size = len(self.image_name_list)
        self.transform = transform
    def __getitem__(self, index):
        sample = {}
        image = tifffile.imread(self.image_name_list[index])
        mask = tifffile.imread(self.mask_name_list[index])  # Y X
        sample['image'] = torch.from_numpy(image[np.newaxis, ...]).float()
        class_map, instance_map = self.convert_instance_to_class_ids(mask)
        sample['semantic_mask'] = torch.from_numpy(class_map[np.newaxis, ...])  # 1 Y X
        sample['instance_mask'] = torch.from_numpy(instance_map[np.newaxis, ...]).byte() # 1 Y X
        if(self.transform is not None):
           return self.transform(sample)
        else:
           return sample

    def __len__(self):
        return self.real_size if self.size is None else self.size


    @classmethod
    def convert_instance_to_class_ids(cls, pic, bg_id=0):
        class_map = convert_to_class_labels(pic)
        instance_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.int16)
        mask_fg = pic > bg_id
        if mask_fg.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask_fg])
            instance_map[mask_fg] = ids
        return class_map, instance_map


# get paths to train images and masks
base_dir = '/home/manan/Desktop/EmbedSeg_MIDL/EmbedSeg/examples/2d/dsb-2018/crops/dsb-2018'
train_image_dir, train_mask_dir = base_dir + '/train/images/', base_dir + '/train/masks/'
# get paths to val images and masks
val_image_dir, val_mask_dir = base_dir + '/val/images/', base_dir + '/val/masks/'

# create transform



# create dataset
train_dataset = CustomDataset(train_image_dir, train_mask_dir, size=3000)
val_dataset = CustomDataset(val_image_dir, val_mask_dir, size=3000)

# create device
device = torch.device("cuda:0")

# create dataloader
train_dataset_iterator = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
val_dataset_iterator = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

# create model
#model = UNet(number_of_class=3, number_of_steps=3)
model = UNet(depth = 3, num_classes = 3)
model = torch.nn.DataParallel(model).to(device)

# create criterion
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 5.0]).cuda())  # apply sigmoid, then softmax
criterion = torch.nn.DataParallel(criterion).to(device)

# create optimizer + scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)


def lambda_(epoch):
    return pow((1 - ((epoch) / 200)), 0.9)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_)

def save_checkpoint(state, is_best, save_dir, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(save_dir, name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            save_dir, 'best_iou_model.pth'))

resume_path = os.environ.get('resume_path')
if resume_path is not None and os.path.exists(resume_path):
    print('Resuming model from {}'.format(resume_path))
    state = torch.load(resume_path)
    start_epoch = state['epoch'] + 1
    best_ap = state['best_average_precision']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    


best_average_precision = 0.0
for epoch in tqdm(range(200)):
    scheduler.step(epoch)
    print('='*50)
    print('===> train loss: {:.2f}'.format(train()))
    val_loss, val_average_precision = val()
    print('===> val loss: {:.2f}, average_precision: {:.2f}'.format(val_loss, val_average_precision))
    is_best = val_average_precision > best_average_precision
    best_average_precision = max(val_average_precision, best_average_precision)
    state = {
        'epoch': epoch,
        'best_average_precision': best_average_precision,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }
    save_checkpoint(state, is_best, save_dir='./exp/')
