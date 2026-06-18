### Segmentation using ImageJ macros

**Do this one time only:**  
1. Download [miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already.
    * Follow installation instructions online. **ACCEPT THE DEFAULTS WHEN PROMPTED.** [Email Eeshaan](mailto:er479@cornell.edu) with any questions.
        * If you are on an M1/M2 Mac, download this: [https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh). Then, enter in your terminal `bash Miniconda3-latest-MacOSX-arm64.sh`.
        * If you are on an older Intel Mac, download this: [https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh). Then, enter in your terminal `bash Miniconda3-latest-MacOSX-arm64.sh`.
        * If you are on a Windows computer, download this: [https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe). Double-click the installed file (...exe) and follow the on-screen instructions. I'm not 100% on how the install and future steps work with Windows so please email me so we can meet and get it set up.  
    * Test it by going to your terminal and typing in: `conda`.
2. Clone the repository:
    * `git clone https://github.com/eeshaanrehani/imageJ-tracing.git`
3. Enter the git repository: `cd /path/to/imageJ-tracing/`.
4. Set up conda environment.
    * `conda env create -f environment.yml`
    * If prompted `[y/n]`: `y`.

**Convert dicom stacks to .tif image slices:**
1. Type in your terminal: `conda activate dicom_imagej`.
2. Type in your terminal: `python dcm_to_tiff.py {DICOM_path} {save_directory} {ending_frame_number}`. Replace `{DICOM_path}` and `{save_directory}` with the path to a DICOM file and the directory you want masks and slices to be saved in, and `{ending_frame_number}` with the last frame of the first loop (if there are multiple loops in the scan). This command will convert the DICOM to tiff images for each individual slice, and save the slices as `.../{save_directory}/{DICOM_name}/{DICOM_name}_{slice}.tif`. It will also create another folder inside your save directory, `.../{save_directory}/mask_{DICOM_name}/`, which is where the masks will be saved. 

**Go to FIJI/ImageJ now and follow the SOP on Box to contour ovaries/follicles.**