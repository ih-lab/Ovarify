// ImageJ macro to convert an ultrasound DICOM to a TIFF file and create masks of any given regions in the image
// Eeshaan Rehani - eer4001@med.cornell.edu

// Prompt the user to open a dicom file stored locally and select the save path for slices + masks

print("DICOM path:")
path = File.openDialog("Select DICOM file");
print(path);

print("Save path:");
outputRoot = getDirectory("Select the results directory to save slices and masks in.");
print(outputRoot);

print("Enter in the terminal: conda activate dicom_imagej");
print("Enter in the terminal (replace {} with correct values): python dcm_to_tiff.py {DICOM path} {save directory}");

// Read in DICOM and save all slices as tif images

fileName = File.getNameWithoutExtension(path);
outputDir = outputRoot + fileName + "/";
maskDir = outputRoot + "mask_" + fileName + "/";

flist = getFileList(outputDir);

for (i=0; i<flist.length-1; i++) {
	for (j=i+1; j<flist.length; j++) {
	
		if (numSort(flist[i], flist[j]) > 0) {
			tempF = flist[i];
			flist[i] = flist[j];
			flist[j] = tempF;
		}
	}
}

// get start/end slice numbers

startSlice = getNumber("Enter the first ovary slice: ", 0);
endSlice = getNumber("Enter the end slice: ", flist.length-1);

// open each individual slice and outline ROIs

for (i=0; i<flist.length; i++) {

	if (i>=startSlice && i<=endSlice && ((i-startSlice)%4==0 || i==endSlice)) {
		open(outputDir + flist[i]);
		origID = getImageID();

		// show the tiff image and prompt the user to fill regions

		run("Overlay Options...", "stroke=white overlay");
		run("ROI Manager...");
		waitForUser("Outline and fill regions in the image. Click OK when done.");
		nROIs = roiManager("count");
		newImage("Mask", "8-bit Black", getWidth(), getHeight(), 1);
		maskID = getImageID();
		selectImage(maskID);

		if (nROIs==0) {
			selectImage(maskID);
			run("Select None");
			saveAs("tiff", maskDir + flist[i]);

		} else {
			for (roi=0; roi<nROIs; roi++) {
				roiManager("select", roi);
				run("Create Mask");
			}
			run("ROI Manager...", "combine");
			roiManager("select", 0)
			run("Create Mask");
			saveAs("tiff", maskDir + flist[i]);
		
		}
		
		selectImage(maskID);
		close();
		selectImage(origID);
		close();
		roiManager("reset")

	} else if (i>=startSlice && i<=endSlice) {
		continue;

	} else {
		open(outputDir + flist[i]);
		origID = getImageID();
		newImage("Mask", "8-bit black", getWidth(), getHeight(), 1);
		blankMaskID = getImageID();
		selectImage(blankMaskID);
		run("Select None");
		saveAs("tiff", maskDir + flist[i]);
		close();
		selectImage(origID);
		close();

	}
}

close("*");
print("-------------");
print("DONE");
print("You can close any remaining image/ROI windows now.");
print("RESTART MACRO FOR NEW DICOM");

function numSort(a, b) {
	aNumber = parseInt(a.substring(a.lastIndexOf("_")+1, a.lastIndexOf(".")));
	bNumber = parseInt(b.substring(b.lastIndexOf("_")+1, b.lastIndexOf(".")));
	
	return aNumber - bNumber;
}
