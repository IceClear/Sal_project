import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

HOME_DIR = '/home/iceclear/project_env/salgan'

# Path to SALICON raw data
pathToImages = '../../salicon/images'
pathToMapsTrain = '../../salicon/maps/train'
pathToMapsVal = '../../salicon/maps/val'
#pathToImages = '/home/bat/salgan/images_test'
#pathToMaps = '/home/bat/salgan/maps_test'

# Path to processed data
pathToResizedImagesTrain = '../../salicon/images256x192_train'
pathToResizedMapsTrain = '../../salicon/maps256x192_train'

pathToResizedImagesVal = '../../salicon/images256x192_val'
pathToResizedMapsVal = '../../salicon/maps256x192_val'

pathToResizedImagesTest = '../../salicon/images256x192_test'

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

list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*train*'))]
print(len(list_img_files))
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(pathToImages, curr_file + '.jpg')
    try:
        imageResized = cv2.resize(cv2.imread(full_img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

        full_map_path = os.path.join(pathToMapsTrain, curr_file + '.png')
        mapResized = cv2.resize(cv2.imread(full_map_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(pathToResizedImagesTrain, curr_file + '.png'), imageResized)
        cv2.imwrite(os.path.join(pathToResizedMapsTrain, curr_file + '.png'), mapResized)
    except:
        print('Error')
    #print('Written image: ', pathToResizedImages, curr_file, ' with size = ', imageResized.shape)
    #print('Written map: ', pathToMaps, curr_file, ' with size = ', mapResized.shape)
list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*val*'))]
print(len(list_img_files))
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(pathToImages, curr_file + '.jpg')
    imageResized = cv2.resize(cv2.imread(full_img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

    full_map_path = os.path.join(pathToMapsVal, curr_file + '.png')
    mapResized = cv2.resize(cv2.imread(full_map_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(pathToResizedImagesVal, curr_file + '.png'), imageResized)
    cv2.imwrite(os.path.join(pathToResizedMapsVal, curr_file + '.png'), mapResized)
    #print('Written image: ', pathToResizedImages, curr_file, ' with size = ', imageResized.shape)
    #print('Written map: ', pathToMaps, curr_file, ' with size = ', mapResized.shape)

list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*test*'))]
print(len(list_img_files))
for curr_file in tqdm(list_img_files):
    full_img_path = os.path.join(pathToImages, curr_file + '.jpg')
    imageResized = cv2.resize(cv2.imread(full_img_path), INPUT_SIZE, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(pathToResizedImagesTest, curr_file + '.png'), imageResized)
    #print('Written image: ', pathToResizedImages, curr_file, ' with size = ', imageResized.shape)
    #print('Written map: ', pathToMaps, curr_file, ' with size = ', mapResized.shape)

print('Done resizing images.')
