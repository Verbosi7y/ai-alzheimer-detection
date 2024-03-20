import os
import json
import numpy as np
import nibabel as nib
from PIL import Image
from scipy import ndimage


def read_json(json_dir):
    json_data = {}

    with open(json_dir, 'r') as json_file:
        json_data = json.load(json_file)

    return json_data


def z_slicer(nifti_dir, output_dir):
    img = nib.load(f"{nifti_dir}")
    img_fdata = img.get_fdata()

    # (x, y, z, channel)
    channel_data = img_fdata[:, :, :, 0] # channel 0, representing grayscale intensity

    # we have 128 Z layers but we only want every 4 layers starting from z = 8 and ending at z = 112
    # we can discard information early and later on
    for i in range(8, 112 + 1, 4):
        # extract slice
        slice_data = channel_data[:, :, i] # (x, y, z); we only need the slice from the z-axis
        
        # normalize to 0-255
        slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
        
        # rotate by 90 degrees
        rotated = ndimage.rotate(slice_data, 90, reshape=False)

        # convert to PIL Image
        slice_image = Image.fromarray(rotated)
        
        
        # create folder for each MRI scans
        if not os.path.exists(f"{output_dir}"):
            os.makedirs(f"{output_dir}")

        # save as grayscale PNG
        imageout_path = f"{output_dir}/{i}.png"
        slice_image.save(imageout_path)


if __name__ == "__main__":
    sample_dir = "sample"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    json_dir = "sample_demographics.json"

    sample_dict = read_json(json_dir)

    img_filename = ""
    for subject_id in sample_dict:
        for mri_id in sample_dict[subject_id]:
            img_dir = f"{sample_dir}/{mri_id}/RAW/mpr-1.nifti.img"

            subject_output = f"{output_dir}/{subject_id}/{mri_id}"

            z_slicer(img_dir, subject_output)

