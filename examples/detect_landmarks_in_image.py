import face_alignment
import matplotlib.pyplot as plt
from skimage import io
import argparse
import os
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch
from torch.autograd import Variable
import cv2
from skimage.draw import polygon_perimeter
from skimage.draw import line
import numpy as np
import random


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
ngpu = int(opt.ngpu)
nc = 3

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
if opt.cuda:
    input = input.cuda()

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=opt.cuda, flip_input=False)

for i, data in enumerate(dataloader, 0):
    real_cpu, _ = data
    batch_size = real_cpu.size(0)
    if opt.cuda:
        real_cpu = real_cpu.cuda()
    input.resize_as_(real_cpu).copy_(real_cpu)
    intputv = Variable(input)
    preds_v,_ = fa.get_landmarks(intputv)

preds = preds_v[0,:,:].data.cpu().numpy()
preds = np.rint(preds).astype(np.int32)

input = io.imread('/home/zeyuan/Desktop/testImages2/train/epoch009_fake_B.png')
input = cv2.resize(input, dsize=(int(opt.imageSize), int(opt.imageSize)),
                        interpolation=cv2.INTER_LINEAR)

#TODO: Make this nice
# fig = plt.figure(figsize=plt.figaspect(.5))
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
#
# col_coords = preds[0:17, 0]
# row_coords = preds[0:17, 1]
# tmp = preds[0:17]
# points = np.array([[910, 641], [206, 632], [696, 488], [458, 485]])
# color = np.random.randint(0,255,(3)).tolist()
cv2.polylines(input, [preds[0:17]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.polylines(input, [preds[17:22]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.polylines(input, [preds[22:27]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.polylines(input, [preds[27:31]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.polylines(input, [preds[31:36]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.polylines(input, [preds[36:42]], 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.polylines(input, [preds[42:48]], 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.polylines(input, [preds[48:60]], 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.polylines(input, [preds[60:68]], 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# rr, cc = polygon_perimeter(row_coords, col_coords, shape=input.shape, clip=False)
# input[rr,cc] = 255
# for i in range(17):
#     rr, cc = line(preds[i, 1], preds[i, 0], preds[i+1, 1], preds[i+1, 0])
#     input[rr,cc] = 255

ax.imshow(input)

# ax.imshow(input)
# ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.axis('off')

# fig.canvas.draw()
# data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
# ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
# ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
# ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
# ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
# ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
# ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
# ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
# ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )
# plt.savefig("./test.png")
# ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
x = 1
