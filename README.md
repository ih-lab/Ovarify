# Ovarify
### A deep learning model for automatic calculation of ovarian morphological features from pelvic ultrasonography
For questions/issues contact:
* Eeshaan Rehani: eer4001@med.cornell.edu
* Iman Hajirasouliha: imh2003@med.cornell.edu

## Environment setup
1. Set up the conda environment - `conda env create -f environment.yml`.
2. Activate the conda environment - `conda activate ovarify`.`

## Training
### Segmentation
`CUDA_VISIBLE_DEVICES={gpu} python train.py {config.yaml}`
* `model_type`: `"segmentation"`.
* `dataloader`: Specify one of `["TrainDataset", "TrainDataset_3channeladjacent"]`.

Other parameter descriptions in `train.py` - see `./skeleton_configs/train.yaml` for more information.
### Classification
`CUDA_VISIBLE_DEVICES={gpu} python train.py {config.yaml}`
* `model_type`: `"classification"`
* `dataloader`: Specify one of `["TestDataset", "TestDataset_3channeladjacent"]`.

Other parameter descriptions in `train.py` - see `./skeleton_configs/train.yaml` for more information.
## Testing (all models)
`CUDA_VISIBLE_DEVICES={gpu} python test.py {config.yaml}`
* `model_type`: Specify one of any options above.
* `dataloader`: Specify one of any options above.
* `weights_path`: Specify .pth file that contains trained model weights.

Other parameter descriptions in `test.py` - see `./skeleton_configs/test.yaml` for more information.

## Calculating ovarian volume
`python calc_volume.py {root_folder} {cf_spreadsheet} {save_path}`
* `{root_folder}`: output of ovarian segmentation

## Counting follicles
`python count_follicles.py {root_folder} {save_path}`
* `{root_folder}`: output of follicle segmentation

## How do I structure my directories for training?
| File tree \
| \
| ---- root_dir _(This should consist of only ovarian masks or follicle masks)_ \
| ---- | ---- {Ultrasound}\_{Frame} \
| ---- | ---- | ---- {Ultrasound}\_{Frame}\_orignal.tif/jpg/png/etc. \
| ---- | ---- | ---- {Ultrasound}\_{Frame}\_mask.tif/jpg.png/etc. _(This is optional; only necessary for training)_
