import os.path
import torch
import argparse
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


#helper functions
#https://gist.github.com/andrewgiessel/4635563
def makeGaussian(self, size, fwhm=3, center=None):
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

for i in range(n_images_csv):
    img_name = keypoints_frame.iloc[i, 0]

    keypoints = keypoints_frame.iloc[i, 1:137].as_matrix()
    keypoints = keypoints.astype('float').reshape(-1, 2)
    heatmap = torch.zeros(1024, 1024)                           #hard code the imagesize to be 1024
    if len(gpu_ids) > 0:
        heatmap = heatmap.cuda(gpu_ids[0])
    keypoints = keypoints.clamp(0.0, 1024 - 1).long()            #discard points outside the image
    for i in range(68):
        heatmap[keypoints[i, 1], keypoints[i, 0]] = 1.0          #draw the keypoints on the map; keypoint format: (x, y) -> (col, row)
    heatmap=Variable(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_gaussian = F.conv2d(heatmap,kernel,padding=3)




