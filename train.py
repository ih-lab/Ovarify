
import os
import datetime
from utils import *
import transforms as t
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
import segmentation as s
import classification as c
import argparse
import yaml
from torchinfo import summary


print("Library import complete")

VALID_SCHEDULERS = ['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'CosineWarmupConstantLR', 'CosineWarmupExponentialDecay', None]
VALID_DATALOADERS = ['TrainDataset', 'TrainDataset_3channeladjacent']

parser = argparse.ArgumentParser(description='Train Ovarify Models')
parser.add_argument("yaml_file", type=str, help="path to yaml file")
args = parser.parse_args()

yaml_file = args.yaml_file

config = {}
try:
    config = yaml.safe_load(open(yaml_file, 'r'))
except yaml.YAMLError as exc:
    print("Error in configuration file:", exc)

train_path = config['train_path'] if 'train_path' in config else "/path/to/train/images"
val_path = config['val_path'] if 'val_path' in config else "/path/to/val/images"
save_path = config['save_path'] if 'save_path' in config else "/path/to/save_dir"
dataloader = config['dataloader']
prefetch_factor = config['prefetch_factor'] if 'prefetch_factor' in config else 2
pin_memory = config['pin_memory'] if 'pin_memory' in config else False
persistent_workers = config['persistent_workers'] if 'persistent_workers' in config else False
num_workers = config['num_workers'] if 'num_workers' in config else 4
learning_rate = config['learning_rate'] if 'learning_rate' in config else 0.001
momentum = config['momentum'] if 'momentum' in config else 1
weight_decay = config['weight_decay'] if 'weight_decay' in config else 0.01
num_epochs = config['num_epochs'] if 'num_epochs' in config else 250
batch_size = config['batch_size'] if 'batch_size' in config else 32
nesterov = config['nesterov'] if 'nesterov' in config else False
checkpoint_path = config['checkpoint_path'] if 'checkpoint_path' in config else None
scheduler = config['scheduler'] if 'scheduler' in config else None
loss = config['loss'] if 'loss' in config else 'bce'
model_type = config['model_type']
optimizer = config['optimizer'] if 'optimizer' in config else 'SGD'
input_dim = config['img_size'] if 'img_size' in config else 512
show_arch = config['show_arch'] if 'show_arch' in config else False

if scheduler not in VALID_SCHEDULERS:
    raise ValueError("Scheduler not recognized. Must be one of: {}".format(VALID_SCHEDULERS))
if dataloader not in VALID_DATALOADERS:
    raise ValueError("Dataloader not recognized. Must be one of: {}".format(VALID_DATALOADERS))

# scheduler params:
if scheduler == 'StepLR':
    step_size = config['step_size'] if 'step_size' in config else 20
    gamma = config['gamma'] if 'gamma' in config else 0.5
elif scheduler == 'ReduceLROnPlateau':
    factor = config['factor']
    patience = config['patience']
elif scheduler == 'CosineAnnealingWarmRestarts':
    t_0 = config['T_0']
    t_mult = config['T_mult']
elif scheduler == 'CosineWarmupConstantLR':
    warmup_steps = config['warmup_steps']
elif scheduler == 'CosineWarmupExponentialDecay':
    warmup_steps = config['warmup_steps']
    gamma = config['gamma']
    cooldown_start = config['cooldown_start']

print('-' * 10)
print(yaml.dump(config, sort_keys=False))
print('-' * 10)

if not os.path.isdir(train_path):
    raise FileNotFoundError("Train path does not exist")
if not os.path.isdir(val_path):
    raise FileNotFoundError("Val path does not exist")
if checkpoint_path is not None and not os.path.exists(checkpoint_path):
    raise FileNotFoundError("Checkpoint file does not exist")

print("Paths validated")

seed_everything(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if dataloader == 'TrainDataset_3channeladjacent':
    train_set = TrainDataset_3channeladjacent(train_path, t.get_augmentation(train=True), input_dim, t.get_transform(train=True))
    val_set = TrainDataset_3channeladjacent(val_path, None, input_dim, t.get_transform(train=False))
elif dataloader == 'TrainDataset': 
    train_set = TrainDataset(train_path, t.get_augmentation(train=True), input_dim, t.get_transform(train=True))
    val_set = TrainDataset(val_path, None, input_dim, t.get_transform(train=False))

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory, persistent_workers=persistent_workers),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory, persistent_workers=persistent_workers)
}

print("DataLoader initialized, {} workers".format(num_workers))

aux_params=dict(
    pooling='avg', 
    activation=None, 
    classes=1
)

in_channels = 3

if model_type == 'segmentation':
    model = smp.MAnet('tu-xception41', encoder_weights='imagenet', classes=1, in_channels=in_channels, activation=None) 
elif model_type == 'classification':
    model = models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT')
    model.classifier = torch.nn.Linear(1408, 1)
else:
    raise ValueError("Model type not recognized")

if show_arch:
    summary(model, input_size=(batch_size, 3, 224, 224), depth=5, col_names=["output_size", "num_params", "trainable"])


print('-' * 10)

params = [p for p in model.parameters() if p.requires_grad]

if optimizer == 'AdamW':
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
elif optimizer == 'SGD':
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

if scheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
elif scheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=False, eps=1e-9)
elif scheduler == 'CosineAnnealingWarmRestarts':
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, verbose=False, eta_min=1e-9)
elif scheduler == 'OneCycleLR':
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(dataloaders['train']), epochs=num_epochs)
elif scheduler == 'CosineWarmupConstantLR':
    lr_scheduler = CosineWarmupConstantLR(optimizer, warmup_steps=warmup_steps, eta_min=1e-9)
elif scheduler == 'CosineWarmupExponentialDecay':
    lr_scheduler = CosineWarmupExponentialDecay(optimizer, warmup_steps=warmup_steps, cooldown_start=cooldown_start, gamma=gamma, eta_min=1e-9)
elif not scheduler:
    lr_scheduler = None

if checkpoint_path is not None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    stripped_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(stripped_state_dict, strict=False)
    print(f"Checkpoint loaded")

model = model.to(device)
model = nn.DataParallel(model)
if os.environ.get('CUDA_VISIBLE_DEVICES'):
    print(f"Model initialized on {device} (GPU:{os.environ['CUDA_VISIBLE_DEVICES']})")
else:
    model_device = set(param.device for param in model.parameters())
    print(f"Model initialized on {device} (GPU:{','.join([str(device.index) for device in model_device])})")

if model_type == 'segmentation':
    start = datetime.datetime.now()
    metrics = s.train_model(model, optimizer, lr_scheduler, dataloaders, device, save_path, loss, start, num_epochs=num_epochs)
elif model_type == 'classification':
    start = datetime.datetime.now()
    metrics = c.train_model(model, optimizer, lr_scheduler, dataloaders, device, save_path, loss, start, num_epochs=num_epochs)

print('-' * 10)
print(metrics)
print('-' * 10)
print("Done")
