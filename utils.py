import os
import cv2
import random
import copy
import datetime
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt as distance


class TrainDataset(Dataset):
    def __init__(self, root, imaug, img_size, transforms):
        self.root = root
        self.path = os.listdir(root)
        self.masks = []
        self.origs = []
        self.subdirs = []
        self.filenames = []
        self.transforms = transforms
        self.imaug = imaug
        self.img_size = img_size
        for subdir, dirs, files in os.walk(root):
            if any("_original" in f for f in files) and any("_mask" in f for f in files) and not any("_original" in f and "_mask" in f for f in files):
                self.subdirs.append(os.path.basename(os.path.normpath(subdir)))
                for file in files:
                    if 'mask' in file:
                        self.masks.append(os.path.join(subdir, file))
                    elif 'original' in file:
                       self.origs.append(os.path.join(subdir, file))
                       self.filenames.append(file[:-13])
        assert len(self.masks) == len(self.origs)

    def __getitem__(self, idx):
        files = self.filenames[idx]
        dirs = self.subdirs[idx]
        mask_path = self.masks[idx]
        orig_path = self.origs[idx]

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = mask.astype(np.uint8)
        mask = np.where(mask >= 127, 255, 0)

        img8 = cv2.imread(orig_path)
        img8 = img8.astype(np.uint8)
        row_img, col_img = img8.shape[0], img8.shape[1]
        img8 = cv2.resize(img8, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if self.imaug is not None:
            transformed = self.imaug(image = img8, mask=mask)
            img8 = transformed['image']
            mask = transformed['mask']
  
        if np.amax(mask) == 0:
            label = torch.tensor([0]).float()
        else:
            label = torch.tensor([1]).float()

        mask = mask.astype(np.float32)/255
        mask = np.where(mask >= 0.5, 1, 0)

        masks = torch.as_tensor(mask, dtype=torch.float32)
        masks = torch.unsqueeze(torch.squeeze(masks, -1), 0)

        if self.transforms is not None:
            img8, masks = self.transforms(img8, masks)
        
        return img8, masks, label, dirs, files, row_img, col_img

    def __len__(self):
        return len(self.origs)

class TestDataset(Dataset):
    def __init__(self, root, img_size, transforms):
        self.root = root
        self.path = os.listdir(root)
        self.origs = []
        self.subdirs = []
        self.filenames = []
        self.transforms = transforms
        self.img_size = img_size
        for subdir, dirs, files in os.walk(root):
            if any("_original" in f for f in files):
                self.subdirs.append(os.path.basename(os.path.normpath(subdir)))
                for file in files:
                    if 'original' in file:
                       self.origs.append(os.path.join(subdir, file))
                       self.filenames.append(file[:-13])
        assert len(self.origs) > 0

    def __getitem__(self, idx):
        files = self.filenames[idx]
        dirs = self.subdirs[idx]
        orig_path = self.origs[idx]

        img8 = cv2.imread(orig_path)
        img8 = img8.astype(np.uint8)
        row_img, col_img = img8.shape[0], img8.shape[1]
        img8 = cv2.resize(img8, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if self.transforms is not None:
            transformed = self.transforms(image = img8)
            img8 = transformed['image']

        return img8, dirs, files, row_img, col_img
    
    def __len__(self):
        return len(self.origs)

class TrainDataset_3channeladjacent(Dataset):
    def __init__(self, root, imaug, img_size, transforms):
        self.root = root
        self.path = os.listdir(root)
        self.masks = []
        self.origs = []
        self.subdirs = []
        self.filenames = []
        self.transforms = transforms
        self.imaug = imaug
        self.img_size = img_size
        for subdir, dirs, files in os.walk(root):
            if any("_original" in f for f in files) and any("_mask" in f for f in files) and not any("_original" in f and "_mask" in f for f in files):
                self.subdirs.append(os.path.basename(os.path.normpath(subdir)))
                for file in files:
                    if 'mask' in file:
                        self.masks.append(os.path.join(subdir, file))
                    elif 'original' in file:
                       self.origs.append(os.path.join(subdir, file))
                       self.filenames.append(file[:-13])
        assert len(self.masks) == len(self.origs)

    def get_next_frame(self, frames, frame):
        next_frame_idx = frames.index(frame) + 1
        if next_frame_idx < len(frames):
            return frames[next_frame_idx]
        return None
    
    def get_prev_frame(self, frames, frame):
        prev_frame_idx = frames.index(frame) - 1
        if prev_frame_idx >= 0:
            return frames[prev_frame_idx]
        return None

    def __getitem__(self, idx):
        folder_name = self.subdirs[idx]
        parts = self.filenames[idx].split('_')
        frame = int(parts[-1])
        identifier = "_".join(parts[:-1])

        folder_parts = folder_name.split('_')
        folder_identifier = "_".join(folder_parts[:-1])

        mask_path = self.masks[idx]
        orig_path = self.origs[idx]

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = mask.astype(np.uint8)
        mask = np.where(mask >= 127, 255, 0)
        sh = mask.shape

        img8 = cv2.imread(orig_path)

        frames = [int(f.split('_')[-1]) for f in self.subdirs if f.startswith(folder_identifier + "_")]
        frames.sort()

        prev_frame = self.get_prev_frame(frames, frame)
        next_frame = self.get_next_frame(frames, frame)

        if prev_frame:
            prev_frame_path = os.path.join(self.root, folder_identifier + "_" + str(prev_frame), identifier + "_" + str(prev_frame) + "_original.tif")
            img8_prev = cv2.imread(prev_frame_path)
        else:
            img8_prev = img8
        
        if next_frame:
            next_frame_path = os.path.join(self.root, folder_identifier + "_" + str(next_frame), identifier + "_" + str(next_frame) + "_original.tif")
            img8_next = cv2.imread(next_frame_path)
        else:
            img8_next = img8

        img8 = img8.astype(np.uint8)
        row_img, col_img = img8.shape[0], img8.shape[1]
        img8 = cv2.resize(img8, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        img8 = np.expand_dims(img8, axis=2)

        img8_prev = img8_prev.astype(np.uint8)
        img8_prev = cv2.resize(img8_prev, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img8_prev = cv2.cvtColor(img8_prev, cv2.COLOR_BGR2GRAY)
        img8_prev = np.expand_dims(img8_prev, axis=2)

        img8_next = img8_next.astype(np.uint8)
        img8_next = cv2.resize(img8_next, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img8_next = cv2.cvtColor(img8_next, cv2.COLOR_BGR2GRAY)
        img8_next = np.expand_dims(img8_next, axis=2)

        img8 = np.concatenate((img8_prev, img8, img8_next), axis=2)

        if self.imaug is not None:
            transformed = self.imaug(image=img8, mask=mask)
            img8 = transformed['image']
            mask = transformed['mask']

        if np.amax(mask) == 0:
            label = torch.tensor([0]).float()
        else:
            label = torch.tensor([1]).float()

        mask = mask.astype(np.float32)/255
        mask = np.where(mask >= 0.5, 1, 0)
    
        masks = torch.as_tensor(mask, dtype=torch.float32)
        masks = torch.unsqueeze(torch.squeeze(masks, -1), 0)

        if self.transforms is not None:
            img8, masks = self.transforms(img8, masks)

        return img8, masks, label, folder_name, self.filenames[idx], row_img, col_img

    def __len__(self):
        return len(self.origs)

class TestDataset_3channeladjacent(Dataset):
    def __init__(self, root, img_size, transforms):
        self.root = root
        self.path = os.listdir(root)
        self.origs = []
        self.subdirs = []
        self.filenames = []
        self.transforms = transforms
        self.img_size = img_size
        for subdir, dirs, files in os.walk(root):
            if any("_original" in f for f in files):
                self.subdirs.append(os.path.basename(os.path.normpath(subdir)))
                for file in files:
                    if 'original' in file:
                       self.origs.append(os.path.join(subdir, file))
                       self.filenames.append(file[:-13])
        assert len(self.origs) > 0
    
    def get_next_frame(self, frames, frame):
        next_frame_idx = frames.index(frame) + 1
        if next_frame_idx < len(frames):
            return frames[next_frame_idx]
        return None
    
    def get_prev_frame(self, frames, frame):
        prev_frame_idx = frames.index(frame) - 1
        if prev_frame_idx >= 0:
            return frames[prev_frame_idx]
        return None
    
    def __getitem__(self, idx):
        folder_name = self.subdirs[idx]
        parts = self.filenames[idx].split('_')
        frame = int(parts[-1])
        identifier = "_".join(parts[:-1])

        folder_parts = folder_name.split('_')
        folder_identifier = "_".join(folder_parts[:-1])

        orig_path = self.origs[idx]

        img8 = cv2.imread(orig_path)

        frames = [int(f.split('_')[-1]) for f in self.subdirs if f.startswith(folder_identifier + "_")]
        frames.sort()

        prev_frame = self.get_prev_frame(frames, frame)
        next_frame = self.get_next_frame(frames, frame)

        if prev_frame:
            prev_frame_path = os.path.join(self.root, folder_identifier + "_" + str(prev_frame), identifier + "_" + str(prev_frame) + "_original.tif")
            img8_prev = cv2.imread(prev_frame_path)
        else:
            img8_prev = img8
        
        if next_frame:
            next_frame_path = os.path.join(self.root, folder_identifier + "_" + str(next_frame), identifier + "_" + str(next_frame) + "_original.tif")
            img8_next = cv2.imread(next_frame_path)
        else:
            img8_next = img8

        img8 = img8.astype(np.uint8)
        row_img, col_img = img8.shape[0], img8.shape[1]
        img8 = cv2.resize(img8, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        img8 = np.expand_dims(img8, axis=2)

        img8_prev = img8_prev.astype(np.uint8)
        img8_prev = cv2.resize(img8_prev, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img8_prev = cv2.cvtColor(img8_prev, cv2.COLOR_BGR2GRAY)
        img8_prev = np.expand_dims(img8_prev, axis=2)

        img8_next = img8_next.astype(np.uint8)
        img8_next = cv2.resize(img8_next, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img8_next = cv2.cvtColor(img8_next, cv2.COLOR_BGR2GRAY)
        img8_next = np.expand_dims(img8_next, axis=2)

        img8 = np.concatenate((img8_prev, img8, img8_next), axis=2)

        if self.transforms is not None:
            transformed = self.transforms(image = img8)
            img8 = transformed['image']
        
        return img8, folder_name, self.filenames[idx], row_img, col_img
    
    def __len__(self):
        return len(self.origs)

class CosineWarmupConstantLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, eta_min=1e-9, last_epoch=-1):
        """
        LR scheduler follows cosine warmup schedule from epoch 0 to warmup_steps, then constant LR afterwards. 
        LR range is [eta_min, base_lr]
        """
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 - math.cos(math.pi * self.last_epoch / self.warmup_steps)) for base_lr in self.base_lrs]
        else:
            return [self.base_lrs[0] for _ in self.base_lrs]

class CosineWarmupExponentialDecay(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, cooldown_start, gamma=0.9, eta_min=1e-9, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.cooldown_start = cooldown_start
        self.gamma = gamma
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 - math.cos(math.pi * self.last_epoch / self.warmup_steps)) for base_lr in self.base_lrs]
        elif self.last_epoch < self.cooldown_start:
            return [self.base_lrs[0] for _ in self.base_lrs]
        else:
            return [base_lr * self.gamma ** (self.last_epoch - self.cooldown_start) for base_lr in self.base_lrs]

def convert_time(time_as_timedelta: datetime.timedelta):
    days = time_as_timedelta.days
    hours = time_as_timedelta.seconds // 3600
    minutes = (time_as_timedelta.seconds // 60) % 60
    seconds = time_as_timedelta.seconds % 60
    
    time_as_str = f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"

    return time_as_str

def data_check(inputs, labels, classif, row_img, col_img, files, save_path):
    """
    Sanity check for data loading and preprocessing
    """
    print("inputs shape: {}".format(inputs.shape))
    print("labels shape: {}".format(labels.shape))
    print("classif shape: {}".format(classif.shape))
    print("row_img shape: {}".format(row_img.shape))
    print("col_img shape: {}".format(col_img.shape))

    for j in range(inputs.shape[0]):
        inputs_j = inputs[j,:,:,:]
        print("input max, min: {}".format([torch.max(inputs_j), torch.min(inputs_j)]))
        print("label raw max, min: {}".format([torch.max(labels[j,:,:,:]), torch.min(labels[j,:,:,:])]))
        print("input type: {}".format(inputs_j.dtype))
        inputs_j = (inputs_j.data.cpu().numpy()*255).astype(np.uint8)
        inputs_j = np.transpose(inputs_j, (1,2,0))
        if not os.path.isdir(f'{save_path}/images'):
            os.makedirs(f'{save_path}/images')
        cv2.imwrite(f'{save_path}/images/{files[j]}_input.jpg', inputs_j)
        labels_j = labels[j,:,:,:]
        labels_j2 = copy.deepcopy(labels_j)
        print("label unique values: {}".format(np.unique(labels_j2.data.cpu().numpy())))
        labels_j = torch.squeeze(labels_j, 0)
        labels_j = (labels_j.data.cpu().numpy()*255).astype(np.uint8)
        print("label image max, min: {}".format([np.max(labels_j), np.min(labels_j)]))
        cv2.imwrite(f'{save_path}/images/{files[j]}_label.jpg', labels_j)

def seed_everything(seed=42):                                                  
    random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False

def compute_class_weights(dataloaders, device, model_type='seg'):
    pos, neg = 0, 0

    if model_type == 'seg':
        print("Calculating class weights for segmentation")
        for data in tqdm(dataloaders['train'], total=len(dataloaders['train'])):
            labels = data[1]
            neg += (labels==0).sum()
            pos += labels.sum()
    elif model_type == 'class':
        print("Calculating class weights for classification")
        for data in tqdm(dataloaders['train'], total=len(dataloaders['train'])):
            classif = data[2]
            neg += (classif==0).sum()
            pos += classif.sum()
        
    pos_weight = neg/pos
    return pos_weight.to(device)

def compute_sdm(img):
    """
    from: https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py#L64
    paper: https://arxiv.org/pdf/1912.03849.pdf
    """
    normalized_sdf = np.zeros(img.shape)
    for b in range(img.shape[0]): # batch size
        for c in range(img.shape[1]): # channel
            posmask = img[b].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = find_boundaries(posmask, mode='inner').astype('uint8')
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
            
    return normalized_sdf

def compute_iou(mask_gt, mask_pred):
    """
    Compute IoU given two masks (ground truth and predicted) of shape (H, W, 1)
    """
    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)
    if np.sum(union) == 0:
        iou = 1 if np.sum(intersection) == 0 else 0
        return iou
    iou = np.sum(intersection)/np.sum(union)

    return iou
