import os
import sys
import pydicom
from PIL import Image

def dcm_to_tiff(dcm_path, save_path, last_frame):

	create_folders(dcm_path, save_path)
	
	dcm_path = os.path.normpath(dcm_path)
	path_components = dcm_path.split(os.sep)
	sample = path_components[-1][:-4]
	ds = pydicom.dcmread(dcm_path, force=True)
	pixels = ds.pixel_array
	if last_frame != 0: 
		pixels = pixels[:last_frame+1]

	for i,img in enumerate(pixels):
		pil_img = Image.fromarray(img)
		r, g, b = pil_img.split()
		g = r.copy()
		b = r.copy()
		fixed_pil_img = Image.merge('RGB', (r, g, b))
		
		width, height = fixed_pil_img.size
		fixed_pil_img = fixed_pil_img.crop((35, 60, width-105, height))
		
		out_img_fname = path_components[-1].replace(".dcm", "_{}.tif".format(i))
		fixed_pil_img.save(os.path.join(save_path, sample, out_img_fname))

		print("Saved slice {} as {}".format(i, out_img_fname))

def create_folders(dcm_path, save_path):
	dcm_name = os.path.splitext(os.path.basename(dcm_path))[0]
	main_folder = os.path.join(save_path, dcm_name)

	if not os.path.exists(main_folder):
		os.makedirs(main_folder)

	mask_folder_name = "mask_" + dcm_name
	mask_folder = os.path.join(save_path, mask_folder_name)

	if not os.path.exists(mask_folder):
		os.makedirs(mask_folder)

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("ERROR: incorrect usage")
		print("Usage: python dcm_to_tiff.py <dcm_path> <save_path>")
		print("Use 0 for ending_frame_number if you want to convert all frames")
	else:
		dcm_path = sys.argv[1]
		save_path = sys.argv[2]
		last_frame = int(sys.argv[3])
		dcm_to_tiff(dcm_path, save_path, last_frame)
