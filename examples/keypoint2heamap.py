import os.path
import torch
import argparse
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import cv2


#helper functions
#https://gist.github.com/andrewgiessel/4635563
def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

#end of helper functions

parser = argparse.ArgumentParser()

parser.add_argument('--keypointroot', required=True, help='path to keypoint csv file')
parser.add_argument('--outf', required=True, help='path to the output directory')
parser.add_argument('--outlow', required=True, help='path to the output directory of 256 images')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

opt = parser.parse_args()
str_ids = opt.gpu_ids.split(',')

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

keypoints_frame = pd.read_csv(opt.keypointroot)

n_images_csv = keypoints_frame.shape[0]

#build the gaussian kernel
g = makeGaussian(size=7, fwhm=3)
kernel = torch.FloatTensor(g)
kernel = torch.stack([kernel for i in range(1)])
kernel = torch.stack([kernel for i in range(1)])    #filter size: out_channel, in_channel, h, w
if len(gpu_ids) > 0:
            kernel = kernel.cuda(gpu_ids[0])
kernel = Variable(kernel)

#to be consistent with the training/testing code, we use torch conv layer as a gaussian filter here
for i in range(n_images_csv):
    print(i)
    img_name = keypoints_frame.iloc[i, 0]

    keypoints = keypoints_frame.iloc[i, 1:137].as_matrix()
    keypoints = torch.from_numpy(keypoints.astype('float').reshape(-1, 2))
    heatmap = torch.zeros(1024, 1024)                           #hard code the imagesize to be 1024
    if len(gpu_ids) > 0:
        heatmap = heatmap.cuda(gpu_ids[0])
        keypoints = keypoints.cuda(gpu_ids[0])
    keypoints = keypoints.clamp(0.0, 1024 - 1).long()            #discard points outside the image
    for i in range(68):
        heatmap[keypoints[i, 1], keypoints[i, 0]] = 1.0          #draw the keypoints on the map; keypoint format: (x, y) -> (col, row)
    heatmap=Variable(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_gaussian = F.conv2d(heatmap,kernel,padding=3).squeeze()
    hm_numpy = (heatmap_gaussian.cpu().data.float().numpy() * 255.0).astype(np.uint8)

    out_path = os.path.join(opt.outf, img_name)
    cv2.imwrite(out_path, hm_numpy)

    #resize to 256 version
    out_path_low = os.path.join(opt.outlow, img_name)
    hm_numpy_low = cv2.resize(hm_numpy, (256, 256))
    cv2.imwrite(out_path_low, hm_numpy_low)

    # src1 = cv2.imread(os.path.join("/media/zeyuan/ACD207B0D2077DB8/dataset/celeba_1024/testB",img_name))
    # src2 =  cv2.cvtColor(hm_numpy,cv2.COLOR_GRAY2RGB)
    # img_blend =cv2.addWeighted( src1, 0.5, src2, 0.5, 0.0)
    # cv2.imshow('img_blend', img_blend)




