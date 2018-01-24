# extracts images and modality from .dcm files
# then resizes all images to the same size

import dicom
import urllib
import os
from numpy import uint8, double, dot
import argparse
import pickle
from scipy

parser = argparse.ArgumentParser(description='extracts images and attributes')
parser.add_argument('-l', '--loc',  help='location of dicom images',
                    const='C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/functions_for_database_CBIR/all_images_sample')
parser.add_argument('-s', '--save',  help='(y/n) save dictionaries?', const='y', required=False)

# returns resized image
def resize_image(img, h_=200, w_=300):
    return imresize(img, (h_, w_))

# convert to gray scale
def gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# returns extracted modality, body part, and pixel array for each dicom image
# in the form of a dictionary: {image_id:(pixel_arr, modality)}
def extract_pixels_and_attributes(dicom_images_loc, resize=True, normalize=True, gray=True):
    # change to directory with images
    os.chdir(dicom_images_loc)
    image_ids_list = os.listdir('./')

    image_pixel_dict = {} # to store pixel arr of each image
    image_modality_dict = {} # to store modality of each image
    image_body_part_dict = {} # to store body part that each image represents
    fail_count = 0 # count of number of images without required attributes

    # iterate through each image in the folder
    for image_id in image_ids_list:

        try:
            dicom_image_temp = dicom.read_file(image_id) # read as dicom
        except: # most likely due to non-dicom hidden files
            print('failed:', image_id)
            continue

        try:
            pixel_array_ = dicom_image_temp.pixel_array
            if resize:
                pixel_array_ = resize_image(pixel_array_)
            if normalize:
                temp_arr = double(pixel_array_)
                pixel_array_ = uint8(255*((temp_arr - temp_arr.min()) / (temp_arr.max() - temp_arr.min())))
            if gray:
                pixel_array_ = gray(rgb)
            image_pixel_dict[image_id] = pixel_array_
            image_modality_dict[image_id] = dicom_image_temp.Modality
            image_body_part_dict[image_id] = dicom_image_temp.BodyPartExamined
        except: # if either or both attributes not found in dicom
            fail_count += 1
            print('Dicom Success, but attribute(s) not found. Image ID: ' + image_id)
            continue

    return fail_count, image_pixel_dict, image_modality_dict, image_body_part_dict


if __name__ = '__main__':
    # read in the args
    args = vars(parser.parse_args())
    loc = args['loc']
    save = args['save']
    # generate image, modality, and body part dicts
    fail_count, image_pixel_dict, image_modality_dict, image_body_part_dict =\
     extract_pixels_and_attributes(dicom_images_loc=loc)
    if save:
        # save all three dicts
        with f as open('image_dict.pickle','wb'): pickle.dump(image_pixel_dict, f)
        with f as open('modality_dict.pickle','wb'): pickle.dump(image_modality_dict, f)
        with f as open('body_part_dict.pickle','wb'): pickle.dump(image_body_part_dict, f)
