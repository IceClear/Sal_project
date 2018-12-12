# %matplotlib inline
import glob
import os
import cv2
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from generator import Generator
from discriminator import Discriminator
from PIL import Image

def to_variable(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

def show(img): # Display rgb tensor image
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)

def show_gray(img): # Display grayscale tensor image
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)

def show_img_from_path(imgPath):
    pilImg = Image.open(imgPath)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)

def predict(model, img):
    to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
    im = to_tensor(img)
    inp = to_variable(im.unsqueeze(0), False)
    out = model(inp)
    map_out = out.cpu().data.squeeze(0)
    return map_out.squeeze(0)

def load_checkpoint(model, optimizer=None, losslogger=None, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger

def calc_cc_score(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def calc_kl_score(gtsAnn, resAnn,eps = 1e-7):
    if np.sum(gtsAnn) > 0:
        gtsAnn = gtsAnn / np.sum(gtsAnn)
    if np.sum(resAnn) > 0:
        resAnn = resAnn / np.sum(resAnn)
    return np.sum(gtsAnn * np.log(eps + gtsAnn / (resAnn + eps)))



pathToResizedImagesTest = '../../Database/images256x192_test/'
pathToResizedMapsTest = '../../Database/3-Saliency-TestSet/All_test_hmap/'
pathToSaveMapsTest = '../../Database/images256x192_result/'

if not os.path.exists(pathToSaveMapsTest):
    os.makedirs(pathToSaveMapsTest)

list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToResizedImagesTest, '*.jpg'))]
print(len(list_img))

model = Generator()
pretrained_dict, _, _, _= load_checkpoint(model, None, None, 'generator.pth.tar')

if torch.cuda.is_available():
    pretrained_dict.cuda()
#print(model)
cc_list = []
kl_list = []
for img_index in list_img:
    imgName = img_index + ".jpg"
    img_path = pathToResizedImagesTest + imgName
    map_ground_truth = pathToResizedMapsTest + imgName
    img = cv2.imread(img_path)
    map_gt = cv2.imread(map_ground_truth)
    map_gt = cv2.resize(map_gt,(img.shape[1],img.shape[0]))
    map_gt = cv2.cvtColor(map_gt, cv2.COLOR_BGR2GRAY)
    sal_predicted = predict(pretrained_dict,img)
    sal_predicted = np.array(sal_predicted)
    sal_predicted = sal_predicted/np.max(sal_predicted)*255
    cv2.imwrite(os.path.join(pathToSaveMapsTest, img_index + '.jpg'), sal_predicted)
    cc_temp = calc_cc_score(map_gt,sal_predicted)
    cc_list += [cc_temp]
    kl_temp = calc_kl_score(map_gt,sal_predicted)
    kl_list += [kl_temp]
cc_avg = np.mean(np.array(cc_list))
kl_avg = np.mean(np.array(kl_list))
print('CC: ', cc_avg)
print('KL: ', kl_avg)
print('Test Done')
