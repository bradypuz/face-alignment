from __future__ import print_function
import os
import sys
import time
import torch
import math
import numpy as np
from torch.autograd import Variable
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2



def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image


def transform(point, center, scale, resolution, invert=False):
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()

def transform_V(point, center, scale, resolution, invert=False):
    _pt = Variable(torch.ones(3))
    t = Variable(torch.eye(3))
    if point.data.is_cuda:
        _pt = _pt.cuda()
        t = t.cuda()
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


def crop(image, center, scale, resolution=256.0, use_cuda=True):
    # Crop around the center point
    """ Crops the image around the center. Input is expected to be an Variable of FloatTensor """
    nc, ht, wd = image.size()
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    newH, newW = br[1] - ul[1], br[0] - ul[0]

    newImg = Variable(torch.zeros(nc, newH, newW))
    if use_cuda:
        newImg = newImg.cuda()

    new_X0, newX1 = int(max(1, -ul[0] + 1)), int(min(br[0], wd) - ul[0])
    new_Y0, newY1 = int(max(1, -ul[1] + 1)), int(min(br[1], ht) - ul[1])
    old_X0, oldX1 = int(max(1, ul[0] + 1)), int(min(br[0], wd))
    old_Y0, oldY1 = int(max(1, ul[1] + 1)), int(min(br[1], ht))
    newImg[:, new_Y0 - 1:newY1, new_X0 - 1:newX1] = image[:, old_Y0 - 1:oldY1, old_X0 - 1:oldX1]
    newImg = newImg.unsqueeze(0)
    newImg = F.upsample_bilinear(newImg, int(resolution))
    # newImg = F.upsample(newImg,int(resolution))
    return newImg


def get_preds_fromhm(hm, centers=None, scales=None):
    max, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    # for i in range()
    preds[..., 0] = preds[...,0] - 1
    preds[..., 0] = torch.fmod(preds[..., 0], hm.size(3)) + 1
    preds[..., 1] = (preds[..., 1] - 1).div(hm.size(2)).floor().add(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                sign_0 = torch.sign(hm_[pY, pX + 1] - hm_[pY, pX - 1])
                sign_1 = torch.sign(hm_[pY + 1, pX] - hm_[pY - 1, pX])
                preds[i, j, 0] = preds[i,j,0].add(sign_0.mul(.25))
                preds[i, j, 1] = preds[i,j,1].add(sign_1.mul(.25))

    preds.add_(-.5)

    preds_orig = Variable(torch.zeros(preds.size()))
    if preds.data.is_cuda:
        preds_orig = preds_orig.cuda()
    if centers is not None and scales is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform_V(
                    preds[i, j], centers[i], scales[i], hm.size(2), True)

    return preds, preds_orig

# From pyzolib/paths.py (https://bitbucket.org/pyzo/pyzolib/src/tip/paths.py)


def appdata_dir(appname=None, roaming=False):
    """ appdata_dir(appname=None, roaming=False)

    Get the path to the application directory, where applications are allowed
    to write user specific files (e.g. configurations). For non-user specific
    data, consider using common_appdata_dir().
    If appname is given, a subdir is appended (and created if necessary).
    If roaming is True, will prefer a roaming directory (Windows Vista/7).
    """

    # Define default user directory
    userDir = os.getenv('FACEALIGNMENT_USERDIR', None)
    if userDir is None:
        userDir = os.path.expanduser('~')
        if not os.path.isdir(userDir):  # pragma: no cover
            userDir = '/var/tmp'  # issue #54

    # Get system app data dir
    path = None
    if sys.platform.startswith('win'):
        path1, path2 = os.getenv('LOCALAPPDATA'), os.getenv('APPDATA')
        path = (path2 or path1) if roaming else (path1 or path2)
    elif sys.platform.startswith('darwin'):
        path = os.path.join(userDir, 'Library', 'Application Support')
    # On Linux and as fallback
    if not (path and os.path.isdir(path)):
        path = userDir

    # Maybe we should store things local to the executable (in case of a
    # portable distro or a frozen application that wants to be portable)
    prefix = sys.prefix
    if getattr(sys, 'frozen', None):
        prefix = os.path.abspath(os.path.dirname(sys.executable))
    for reldir in ('settings', '../settings'):
        localpath = os.path.abspath(os.path.join(prefix, reldir))
        if os.path.isdir(localpath):  # pragma: no cover
            try:
                open(os.path.join(localpath, 'test.write'), 'wb').close()
                os.remove(os.path.join(localpath, 'test.write'))
            except IOError:
                pass  # We cannot write in this directory
            else:
                path = localpath
                break

    # Get path specific for this app
    if appname:
        if path == userDir:
            appname = '.' + appname.lstrip('.')  # Make it a hidden directory
        path = os.path.join(path, appname)
        if not os.path.isdir(path):  # pragma: no cover
            os.mkdir(path)

    # Done
    return path


def shuffle_lr(parts, pairs=None):
    if pairs is None:
        pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10],
                 [7, 9], [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [36, 45],
                 [37, 44], [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34],
                 [50, 52], [49, 53], [48, 54], [61, 63], [60, 64], [67, 65], [59, 55], [58, 56]]
    for matched_p in pairs:
        idx1, idx2 = matched_p[0], matched_p[1]
        tmp = np.copy(parts[..., idx1])
        np.copyto(parts[..., idx1], parts[..., idx2])
        np.copyto(parts[..., idx2], tmp)
    return parts


def flip(tensor, is_label=False):
    was_cuda = False
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    elif isinstance(tensor, torch.cuda.FloatTensor):
        tensor = tensor.cpu().numpy()
        was_cuda = True

    was_squeezed = False
    if tensor.ndim == 4:
        tensor = np.squeeze(tensor)
        was_squeezed = True
    if is_label:
        tensor = tensor.swapaxes(0, 1).swapaxes(1, 2)
        tensor = cv2.flip(shuffle_lr(tensor), 1).reshape(tensor.shape)
        tensor = tensor.swapaxes(2, 1).swapaxes(1, 0)
    else:
        tensor = cv2.flip(tensor, 1).reshape(tensor.shape)
    if was_squeezed:
        tensor = np.expand_dims(tensor, axis=0)
    tensor = torch.from_numpy(tensor)
    if was_cuda:
        tensor = tensor.cuda()
    return tensor
