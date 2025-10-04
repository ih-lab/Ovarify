import os
import argparse
import numpy as np
from utils import *
import pandas as pd
import cv2
from skimage.feature import peak_local_max


print("Library import complete")

def get_args():
    parser = argparse.ArgumentParser(description='Calculate follicle count for all frames in a root folder')
    parser.add_argument("root_folder", type=str, help="path to root folder")
    parser.add_argument("save_path", type=str, help="folder to save follicle count results")
    args = parser.parse_args()

    root_folder = args.root_folder
    save_folder = args.save_path

    return root_folder, save_folder

def calculate_follicle_count(frame):
    """
    Calculate follicle count given a binary mask frame
    """

    coordinates = peak_local_max(frame, min_distance=5)
    follicle_count = len(coordinates)

    return follicle_count

def main():

    root_folder, save_path = get_args()

    follicle_count = pd.DataFrame(columns=['Ultrasound_Frame', 'Count'])

    for folder in tqdm(os.listdir(root_folder)):
        if os.path.isdir(os.path.join(root_folder, folder)):
            frame = folder.split("_")[-1]
            mask_result_fname = folder + "_mask_result.tif"
            mask = cv2.imread(os.path.join(root_folder, folder, mask_result_fname), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            count = calculate_follicle_count(mask)
            follicle_count = pd.concat([follicle_count, pd.DataFrame([[frame, count]], columns=['Ultrasound_Frame', 'Count'])], ignore_index=True)

    follicle_count.to_csv(os.path.join(save_path, "0000_ALL_FOLLICLE_COUNTS.csv"), index=False)

if __name__ == "__main__":
    main()

    print('-' * 10)
    print("Done")
