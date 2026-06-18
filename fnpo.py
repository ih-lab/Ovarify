import os
import argparse
import numpy as np
from tqdm import tqdm
# from utils import *
import pandas as pd
import cv2
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage import binary_dilation, distance_transform_edt


print("Library import complete")

def get_args():
    parser = argparse.ArgumentParser(description="Count follicles in 3D pelvic ultrasonography for all ultrasounds in a folder")
    parser.add_argument("root_folder", type=str, help="path to root folder")
    parser.add_argument("save_folder", type=str, help="folder to save calculated FNPO")
    parser.add_argument("save_prefix", type=str, help="csv save prefix")
    args = parser.parse_args()

    root_folder = args.root_folder
    save_folder = args.save_folder
    save_prefix = args.save_prefix

    return root_folder, save_folder, save_prefix

def count_follicles_3d(frame_array, sorted_frame_numbers, min_distance=10, xy_shift=4, z_connect=2):
    """
    Calculate FNPO and FNPS counts from a 3D ultrasound frame array.
    frame_array: 3D numpy array of ultrasound frames' binary follicle masks
    sorted_frame_numbers: list of sorted frame numbers corresponding to the frames in frame_array
    """

    fnpo = 0
    fnps_counts = []

    n_frames, x_dim, y_dim = frame_array.shape

    centroid_volume = np.zeros_like(frame_array, dtype=bool)

    for i in range(n_frames):
        if not np.any(frame_array[i]):
            fnps_counts.append(0)
            continue

        dist = distance_transform_edt(frame_array[i])
        coordinates = peak_local_max(dist, min_distance=min_distance, labels=frame_array[i].astype(int))
        count = coordinates.shape[0]
        fnps_counts.append(count)

        if count > 0:
            centroid_volume[i, coordinates[:, 0], coordinates[:, 1]] = True

    z_size = 2 * z_connect + 1
    xy_size = 2 * xy_shift + 1
    structure = np.zeros((z_size, xy_size, xy_size))
    structure[:, :, :] = 1

    dilated_centroids = binary_dilation(centroid_volume, structure=structure)

    _, fnpo = label(dilated_centroids, return_num=True, connectivity=2)

    return fnpo, fnps_counts

def main():
    
    root_folder, save_folder, save_prefix = get_args()
    counted_follicles = pd.DataFrame(columns=['Ultrasound', 'FNPO', 'FNPS_counts'])

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    ultrasounds = []

    for folder in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, folder)):
            parts = folder.split("_")
            if len(parts) >= 2:
                identifier = "_".join(parts[:-1])
                if identifier not in ultrasounds:
                    ultrasounds.append(identifier)

    for us in tqdm(ultrasounds):
        frame_data = {}
        for folder in os.listdir(root_folder):
            if os.path.isdir(os.path.join(root_folder, folder)) and us in folder:
                frame = folder.split("_")[-1]
                if frame not in frame_data.keys():
                    mask_result_fname = folder + "_mask_result.tif"
                    frame_data[frame] = cv2.imread(os.path.join(root_folder, folder, mask_result_fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        sorted_frame_numbers = sorted(frame_data.keys(), key=lambda x: int(x))
        frame_array = np.empty((len(sorted_frame_numbers), *frame_data[sorted_frame_numbers[0]].shape))

        for i, frame_number in enumerate(sorted_frame_numbers):
            frame_array[i, :, :] = frame_data[frame_number]

        sorted_frame_numbers = [int(val) for val in sorted_frame_numbers]

        fnpo_pred, fnps_counts = count_follicles_3d(frame_array, sorted_frame_numbers)

        additional_row = pd.DataFrame([[us, fnpo_pred, fnps_counts]], columns=['Ultrasound', 'FNPO', 'FNPS_counts'])
        counted_follicles = pd.concat([counted_follicles, additional_row], ignore_index=True)
        tqdm.write(f"Ultrasound: {us}\tFNPO: {fnpo_pred}\tFNPS counts: {fnps_counts}")

    save_fname = save_prefix + "_ALL_COUNTED_FOLLICLES.csv"
    counted_follicles.to_csv(os.path.join(save_folder, save_fname), index=False)

    return

if __name__ == "__main__":
    main()

    print('-' * 10)
    print("Done")
