import os
import datetime
from utils import *
import transforms as t
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
from segmentation import *
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import pandas as pd


print("Library import complete")

VALID_DATALOADERS = ['TestDataset', 'TestDataset_3channeladjacent']

parser = argparse.ArgumentParser(description='Test Ovarify Models')
parser.add_argument("yaml_file", type=str, help="path to yaml file")
args = parser.parse_args()

yaml_file = args.yaml_file

config = {}
try:
    config = yaml.safe_load(open(yaml_file, 'r'))
except yaml.YAMLError as exc:
    print("Error in configuration file:", exc)

weights_path = config['weights_path']
test_path = config['test_path'] if 'test_path' in config else "/path/to/test/images"
save_path = config['save_path'] if 'save_path' in config else "/path/to/save_dir"
dataloader = config['dataloader'] if 'dataloader' in config else "TestDataset"
loss_type = config['loss'] if 'loss' in config else 'jaccard'
model_type = config['model_type'] if 'model_type' in config else "segmentation"
num_workers = config['num_workers'] if 'num_workers' in config else 4
input_dim = config['img_size'] if 'img_size' in config else 512
batch_size = config['batch_size'] if 'batch_size' in config else 16

if dataloader not in VALID_DATALOADERS:
    raise ValueError("Dataloader not recognized. Must be one of: {}".format(VALID_DATALOADERS))

print('-' * 10)
print(yaml.dump(config, sort_keys=False))
print('-' * 10)

if not os.path.isdir(test_path):
    raise FileNotFoundError("Test path does not exist")
if not os.path.exists(weights_path):
    raise FileNotFoundError("Weights file does not exist")

print("Paths validated")

seed_everything(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if dataloader == 'TestDataset_3channeladjacent':
    dataset = TestDataset_3channeladjacent(test_path, None, input_dim, t.get_transform(train=False))
elif dataloader == 'TestDataset':
    dataset = TestDataset(test_path, None, input_dim, t.get_transform(train=False))

test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print("DataLoader initialized, {} workers".format(num_workers))

start_time = datetime.datetime.now()
in_channels = 3

if model_type=='segmentation':
    model = smp.MAnet('tu-xception41', encoder_weights='imagenet', classes=1, in_channels=in_channels, activation=None)
elif model_type=='classification':
    model = models.efficientnet_b2()
    model.classifier = torch.nn.Linear(1408, 1) 

state_dict = torch.load(weights_path)

stripped_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(stripped_state_dict)
model.to(device)
model.eval()
if os.environ.get('CUDA_VISIBLE_DEVICES'):
    print(f"Model initialized on {device} (GPU:{os.environ['CUDA_VISIBLE_DEVICES']})")
else:
    model_device = set(param.device for param in model.parameters())
    print(f"Model initialized on {device} (GPU:{','.join([str(device.index) for device in model_device])})")

channel = int(in_channels/2) + 1
print("Begin inference")

with torch.no_grad():
    if model_type=='segmentation':
        iou_data = []

        for data_img in tqdm(test_loader):
            inputsv, dirs, files, row_imgs, col_imgs = data_img[0].to(device), data_img[1], data_img[2], data_img[3], data_img[4]
            outputs = model(inputsv)

            if loss_type != 'shape':   
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.sigmoid(-1500*outputs)

            for j in range(outputs.shape[0]):
                if outputs[j].shape[0] > 0:
                    save_dir = os.path.join(save_path, "images", str(dirs[j]))
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    col_img, row_img = row_imgs[j].data.cpu().numpy(), col_imgs[j].data.cpu().numpy() 
                    mask_result = outputs[j,:,:,:]
                    mask_result = (mask_result.data.cpu().numpy()*255).astype(np.uint8)
                    mask_result = np.transpose(mask_result, (1,2,0))
                    mask_result = (cv2.resize(mask_result, (int(row_img), int(col_img)), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
                    mask_result = cv2.medianBlur(mask_result, 3) 
                    mask_result = np.where(mask_result >= 127, 255, 0).astype(np.uint8) 
                    cv2.imwrite(f'{save_dir}/{files[j]}_mask_result.tif', mask_result)

        test_time = datetime.datetime.now() - start_time
        test_time = convert_time(test_time)
        print('Test time: {}'.format(test_time))
    elif model_type=='classification':
        class_data = []

        for data_img in tqdm(test_loader):
            inputsv, dirs, files, row_imgs, col_imgs = data_img[0].to(device), data_img[1], data_img[2], data_img[3], data_img[4]
            outputs = model(inputsv)
            outputs = torch.sigmoid(outputs)

            for j in range(outputs.shape[0]):
                if outputs[j].shape[0] > 0:
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                
                    class_data.append({
                        "Image": files[j],
                        "pred_class": outputs[j][0].data.cpu().numpy()
                    })
        
        class_data = pd.DataFrame(class_data)
        class_data.to_csv(os.path.join(save_path, "class.csv"), index=False)

        test_time = datetime.datetime.now() - start_time
        test_time = convert_time(test_time)
        print('Test time: {}'.format(test_time))

print("Done")
