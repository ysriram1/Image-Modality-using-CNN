# extracts images and modality from .dcm files
# then resizes all images to the same size

import dicom
import os
from numpy import uint8, double, dot
import argparse
import pickle
from scipy.misc import imresize
import cv2

parser = argparse.ArgumentParser(description='extracts images and attributes')
parser.add_argument('-l', '--loc',  help='location of dicom images',
                    default='C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/functions_for_database_CBIR/all_images_sample')
parser.add_argument('-s', '--save',  help='(y/n) save dictionaries?', default='y', required=False)
parser.add_argument('-v', '--saveloc',  help='location to save dictionaries',
                    default='C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology', required=False)

# returns resized image
def resize_image(img, h_=200, w_=300):
    return imresize(img, (h_, w_))

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
        image_loc = dicom_images_loc + '/' + image_id
        try:
            dicom_image_temp = dicom.read_file(image_loc) # read as dicom
            print('Dicom read success:', image_id)
        except: # most likely due to non-dicom hidden files
            print('Dicom read failed:', image_id)
            continue
        try:
            pixel_array_ = dicom_image_temp.pixel_array
            if resize:
                pixel_array_ = resize_image(pixel_array_)
            if normalize:
                temp_arr = double(pixel_array_)
                pixel_array_ = uint8(255*((temp_arr - temp_arr.min()) / (temp_arr.max() - temp_arr.min())))
            if gray:
                if len(pixel_array_.shape) == 3:
                    # TODO: replace with something better
                    # hacky way of dealing with images that have color channels first
                    if pixel_array_.shape[0] == 3:
                        pixel_array_ = pixel_array_.T
                    pixel_array_= cv2.cvtColor(pixel_array_,cv2.COLOR_BGR2GRAY)
            image_pixel_dict[image_id] = pixel_array_
            image_modality_dict[image_id] = dicom_image_temp.Modality
            image_body_part_dict[image_id] = dicom_image_temp.BodyPartExamined
        except: # if either or both attributes not found in dicom
            fail_count += 1
            print('Dicom Success, but attribute(s) not found. Image ID: ' + image_id)
            continue

    return fail_count, image_pixel_dict, image_modality_dict, image_body_part_dict


if __name__ == '__main__':
    # read in the args
    args = vars(parser.parse_args())
    loc = args['loc']
    save = args['save']
    save_loc = args['saveloc']
    # generate image, modality, and body part dicts
    fail_count, image_pixel_dict, image_modality_dict, image_body_part_dict =\
     extract_pixels_and_attributes(dicom_images_loc=loc)
    if save == 'y':
        # save all three dicts
        os.chdir(save_loc)
        with open('image_dict.pickle','wb') as f: pickle.dump(image_pixel_dict, f)
        with open('modality_dict.pickle','wb') as f: pickle.dump(image_modality_dict, f)
        with open('body_part_dict.pickle','wb') as f: pickle.dump(image_body_part_dict, f)
