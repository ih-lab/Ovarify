import torchvision_transforms as T
import torch
import torchvision
import albumentations as A


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def get_transform(train):
    transform_list = []
    if train:
        transform_list.append(T.ToTensor())
        transform_list.append(T.Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) # imagenet mean, std
    else:
        transform_list.append(T.ToTensor())
        transform_list.append(T.Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) # imagenet mean, std
    return T.Compose(transform_list)

def get_augmentation(train):
    if train:
        augmentation = A.Compose([
            A.SomeOf([
                A.OneOf([
                    A.HorizontalFlip(p=0.9), 
                    A.VerticalFlip(p=0.9)
                ]), 
                A.OneOf([
                    A.Affine(rotate=90), 
                    A.Affine(rotate=180), 
                    A.Affine(rotate=270)
                ], p=2/3), 
                A.OneOf([
                    A.GridDistortion(p=0.5),
                    A.ElasticTransform(p=0.5),
                ], p=0.9), 
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5), 
                    A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.5),
                ], p=0.5)
            ], n=3)
        ])
        return augmentation
    else:
        return None