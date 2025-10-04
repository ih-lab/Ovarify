import os
import argparse
import numpy as np
from utils import *
import pandas as pd
import cv2


print("Library import complete")

def get_args():
    parser = argparse.ArgumentParser(description='Calculate ovarian volume for all ovaries in a root folder')
    parser.add_argument("root_folder", type=str, help="path to root folder")
    parser.add_argument("cf_spreadsheet", type=str, help="path to cf spreadsheet")
    parser.add_argument("save_path", type=str, help="folder to save calculated volumes")
    args = parser.parse_args()

    root_folder = args.root_folder
    cf_spreadsheet = args.cf_spreadsheet
    save_folder = args.save_path

    return root_folder, cf_spreadsheet, save_folder

def calculate_volume(frames, sorted_frame_numbers, cf, slice_spacing):
    """
    Calculate ovarian volume given:
    cf: conversion factor (mm/pixel)
    sorted_frame_numbers: frame numbers there is a mask for (sorted)
    frames: frame mask in order of sorted_frame_numbers, shape (len(sorted_frame_numbers), image.shape[0], image.shape[1]), numpy.ndarray
    """
    assert frames.shape[0] == len(sorted_frame_numbers)

    slice_areas = pd.DataFrame(columns=['frame_number', 'slice_area_px2', 'slice_area_mm2'])

    for i, frame_number in enumerate(sorted_frame_numbers):
        frame = frames[i]
        frame_area_px2 = np.sum(frame)
        frame_area_mm2 = frame_area_px2 * cf**2
        single_slice = pd.DataFrame([[frame_number, frame_area_px2, frame_area_mm2]], columns=['frame_number', 'slice_area_px2', 'slice_area_mm2'])
        slice_areas = pd.concat([slice_areas, single_slice], axis=0, ignore_index=True)

    slice_areas.sort_values(by=['frame_number'], inplace=True)
    
    interp_frames = []
    for i,frame_number in enumerate(sorted_frame_numbers):
        frame_diff = sorted_frame_numbers[i+1] - frame_number if i < len(sorted_frame_numbers)-1 else 0
        interp_frames.append(frames[i])
        if frame_diff > 1:
            for j in range(1, frame_diff):
                weight_i = (frame_diff - j) / frame_diff
                weight_iplus1 = j / frame_diff
                interpolated_frame = weight_i * frames[i] + weight_iplus1 * frames[i+1]
                interp_frames.append(interpolated_frame)

    interp_frames = np.array(interp_frames)

    volume = np.sum(interp_frames)
    volume_cm3 = volume * (cf**2) * slice_spacing / 1000

    return volume, volume_cm3, slice_areas

def main():

    root_folder, cf_spreadsheet, save_folder = get_args()

    cf_spreadsheet = pd.read_csv(cf_spreadsheet)
    calculated_volumes = pd.DataFrame(columns=['Ultrasound', 'OV_calc_cm3', 'OV_calc_px3'])

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # get all unique ultrasounds and associated frame numbers in root_folder
    ultrasounds = []

    for folder in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, folder)):
            parts = folder.split("_")
            if len(parts) >= 2:
                identifier = "_".join(parts[:-1])
                if identifier not in ultrasounds:
                    ultrasounds.append(identifier)
            
    ultrasounds = [us for us in ultrasounds if us in cf_spreadsheet['Ultrasound'].values]

    for us in tqdm(ultrasounds):
        # get all frames for this ultrasound
        frame_data = {}
        for folder in os.listdir(root_folder):
            if os.path.isdir(os.path.join(root_folder, folder)) and us in folder:
                frame = folder.split("_")[-1]
                if frame not in frame_data.keys():
                    mask_result_fname = folder + "_mask_result.tif"
                    frame_data[frame] = cv2.imread(os.path.join(root_folder, folder, mask_result_fname), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255

        sorted_frame_numbers = sorted(frame_data.keys(), key=lambda x: int(x))
        frame_array = np.empty((len(sorted_frame_numbers), *frame_data[sorted_frame_numbers[0]].shape))

        for i, frame_number in enumerate(sorted_frame_numbers):
            frame_array[i,:,:] = frame_data[frame_number]

        sorted_frame_numbers = [int(val) for val in sorted_frame_numbers]

        cf = float(cf_spreadsheet.loc[cf_spreadsheet['Ultrasound'] == us]['CF'])

        volume_px3, volume_cm3, slice_areas = calculate_volume(frame_array, sorted_frame_numbers, cf, 0.5)

        us_slices_fname = os.path.join(save_folder, f"{us}_slices.csv")
        slice_areas.to_csv(us_slices_fname, index=False)

        volume_row = pd.DataFrame([[us, cf, volume_cm3, volume_px3]], columns=['Ultrasound', 'CF', 'OV_calc_cm3', 'OV_calc_px3'])
        calculated_volumes = pd.concat([calculated_volumes, volume_row], ignore_index=True)
        tqdm.write("Ultrasound: {}\tOV_calc_cm3: {}\tOV_calc_px3: {}".format(us, volume_cm3, volume_px3))

    calculated_volumes.to_csv(os.path.join(save_folder, "0000_ALL_CALCULATED_VOLUMES.csv"), index=False)

if __name__ == "__main__":
    main()

    print('-' * 10)
    print("Done")
