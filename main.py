#!/usr/bin/python

import associate as ass
import matching_patches2 as mp
import os

##### MAKE CHANGES HERE ONLY ##########################
dataset_sequence = 'rgbd_dataset_freiburg2_rpy/'
generated_dataset_folder_name = 'dummy_test'
num_folders = 30
########################################################

os.mkdir(generated_dataset_folder_name)
generated_dataset_path = generated_dataset_folder_name + '/'
dataset_path = '/mnt/data/tum_rgbd_slam/'+dataset_sequence


#generate the 'rgb vs camera_orientation' pairs
rgb_txt_file = dataset_path + 'rgb.txt'
gt_txt_file = dataset_path + 'groundtruth.txt'
output_file = 'rgb_groundtruth.txt'
print "Generating rgb_groundtruth.txt ..."
ass.associate_main(rgb_txt_file,gt_txt_file,generated_dataset_path,output_file)


#generate the 'rgb vs depth' pairs
depth_file = dataset_path + 'depth.txt'
output_file = 'rgb_depth.txt'
print "Generating rgb_depth.txt ..."
ass.associate_main(rgb_txt_file,depth_file,generated_dataset_path,output_file)



#initiate the matching_pairs script to generate data
mp.matching_patches(dataset_sequence,generated_dataset_path,num_folders)
