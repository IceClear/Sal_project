import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

HOME_DIR = '/home/iceclear/project_env/salgan'

# Path to SALICON raw data
pathToImagesTrain = '../../Database/3-Saliency-TrainSet/All_train_images'
pathToImagesTest = '../../Database/3-Saliency-TestSet/All_test_images'
pathToImagesVal = '../../salicon/images'
pathToMapsTrain = '../../Database/3-Saliency-TrainSet/All_train_hmap'
pathToMapsVal = '../../salicon/maps/val'
#pathToImages = '/home/bat/salgan/images_test'
#pathToMaps = '/home/bat/salgan/maps_test'

# Path to processed data
pathToResizedImagesTrain = '../../Database/images256x192_train'
pathToResizedMapsTrain = '../../Database/maps256x192_train'
pathToResizedImagesVal = '../../Database/images256x192_val'
pathToResizedMapsVal = '../../Database/maps256x192_val'
pathToResizedImagesTest = '../../Database/images256x192_test'

INPUT_SIZE = (256, 192)

if not os.path.exists(pathToResizedImagesVal):
    os.makedirs(pathToResizedImagesVal)
if not os.path.exists(pathToResizedMapsVal):
    os.makedirs(pathToResizedMapsVal)
if not os.path.exists(pathToResizedImagesTrain):
    os.makedirs(pathToResizedImagesTrain)
if not os.path.exists(pathToResizedMapsTrain):
    os.makedirs(pathToResizedMapsTrain)
if not os.path.exists(pathToResizedImagesTest):
    os.makedirs(pathToResizedImagesTest)

list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImagesTrain,'*jpg'))]
print(len(list_img_files))
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(pathToImagesTrain, curr_file + '.jpg')
    try:
        imageResized = cv2.resize(cv2.imread(full_img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

        full_map_path = os.path.join(pathToMapsTrain, curr_file + '.jpg')
        mapResized = cv2.resize(cv2.imread(full_map_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(pathToResizedImagesTrain, curr_file + '.png'), imageResized)
        cv2.imwrite(os.path.join(pathToResizedMapsTrain, curr_file + '.png'), mapResized)
    except:
        print('Error')
    #print('Written image: ', pathToResizedImages, curr_file, ' with size = ', imageResized.shape)
    #print('Written map: ', pathToMaps, curr_file, ' with size = ', mapResized.shape)
list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImagesVal, '*val*'))]
print(len(list_img_files))
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(pathToImagesVal, curr_file + '.jpg')
    imageResized = cv2.resize(cv2.imread(full_img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

    full_map_path = os.path.join(pathToMapsVal, curr_file + '.png')
    mapResized = cv2.resize(cv2.imread(full_map_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(pathToResizedImagesVal, curr_file + '.png'), imageResized)
    cv2.imwrite(os.path.join(pathToResizedMapsVal, curr_file + '.png'), mapResized)
    #print('Written image: ', pathToResizedImages, curr_file, ' with size = ', imageResized.shape)
    #print('Written map: ', pathToMaps, curr_file, ' with size = ', mapResized.shape)

list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImagesTest,'*jpg'))]
print(len(list_img_files))
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(pathToImagesTest, curr_file + '.jpg')
    imageResized = cv2.resize(cv2.imread(full_img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(pathToResizedImagesTest, curr_file + '.jpg'), imageResized)
    #print('Written image: ', pathToResizedImages, curr_file, ' with size = ', imageResized.shape)
    #print('Written map: ', pathToMaps, curr_file, ' with size = ', mapResized.shape)

print('Done resizing images.')
