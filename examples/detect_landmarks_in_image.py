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
import pandas as pd
from examples.FaceLandmarksDataset import FaceLandmarksDataset, RandomCrop, Rescale, ToTensor
import random
import torchvision.utils as vision


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--imageSize', type=int, default=1024, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--gpu_ids', default='0', help='the indices of GPUs to use')


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


plt.ion()  # interactive mode

opt = parser.parse_args()
print(opt)
gpus = opt.gpu_ids.split(',')
gpu_ids= []
for i in range(len(gpus)):
    gpu_ids.append(int(gpus[i]))
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
    input = input.cuda(device=gpu_ids[0])

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=opt.cuda, flip_input=False, gpu_ids=gpu_ids)

paths = dataloader.dataset.imgs
num = len(paths)
vals = np.empty([num, 139]) #68*2 + 2 + 1   landmarks:x,y; center: x,y; sclae: 1

cnt = 0

fnames = []
indices = []
#paths -> image names
for i in range(num):
    tmp = os.path.basename(paths[i][0])
    idx = tmp.split('.')[0]
    fnames.append(tmp)
    indices.append(idx)


pth_dir = os.path.join(opt.outf, 'heatmaps')
img_dir = os.path.join(opt.outf, 'imgs')
img_dir_lowRes = os.path.join(opt.outf, 'imgs_256')

for i, data in enumerate(dataloader, 0):
    print("count:%d" % (cnt))
    real_cpu, _ = data
    batch_size = real_cpu.size(0)
    if opt.cuda:
        real_cpu = real_cpu.cuda(device=gpu_ids[0])
    input.resize_as_(real_cpu).copy_(real_cpu)
    intputv = Variable(input)
    preds_v, heatmaps, centers, scales = fa.get_landmarks(intputv)
    preds_v_flat = preds_v.view(batch_size, -1)
    p_np = preds_v_flat.data.cpu().numpy()
    c_np = np.empty([batch_size, 2])
    for i in range(batch_size):
        cur_c = centers[i].cpu().numpy()
        c_np[i, :] = cur_c
    s_np = np.asarray(scales).reshape((batch_size, 1))
    vals[cnt: cnt+batch_size] = np.append(np.append(p_np, c_np, axis=1), s_np, axis=1)


    if not os.path.exists(pth_dir):
        os.makedirs(pth_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(img_dir_lowRes):
        os.makedirs(img_dir_lowRes)
    #save heatmaps and draw connected points as images
    for j in range(batch_size):
        hmap = heatmaps[-1][j, :, :].unsqueeze(0)
        cur_name = indices[cnt+j] + '.pth'
        cur_img_name = indices[cnt+j] + '.png'
        cur_path = os.path.join(pth_dir, cur_name)
        cur_img_path = os.path.join(img_dir, cur_img_name)
        #normalize the heatmaps to [0,1] channel wise
        min, max = torch.min(hmap), torch.max(hmap)
        print("batch:%d, min:%.4f, max:%.4f" % (j, min.data[0], max.data[0]))
        hmap = hmap.cpu().data
        torch.save(hmap, cur_path)

        preds = preds_v[j,:,:].data.cpu().numpy()
        preds = np.rint(preds).astype(np.int32)
        cur_img = np.zeros((opt.imageSize, opt.imageSize, 3))

        cv2.polylines(cur_img, [preds[0:17]], 0, (1, 1, 1), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(cur_img, [preds[17:22]], 0, (2, 2, 2), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(cur_img, [preds[22:27]], 0, (3, 3, 3), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(cur_img, [preds[27:31]], 0, (4, 4, 4), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(cur_img, [preds[31:36]], 0, (5, 5, 5), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(cur_img, [preds[36:42]], 1, (6, 6, 6), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(cur_img, [preds[42:48]], 1, (7, 7, 7), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(cur_img, [preds[48:60]], 1, (8, 8, 8), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(cur_img, [preds[60:68]], 1, (9, 9, 9), thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite(cur_img_path, cur_img)

        cur_img_low_path = os.path.join(img_dir_lowRes, cur_img_name)
        cur_img_lowRes = cv2.resize(cur_img, (256, 256))

        cv2.imwrite(cur_img_low_path, cur_img_lowRes)

    cnt += batch_size

#write the matrix and image names to a single csv file
#----------->x
#|
#|
#|
#y

raw_data = {'names': fnames}
column_names = ['names']
outname = os.path.join(opt.outf,'landmarkds.csv')
#update x y coordinates
for i in range(68):
    cur_x_name = 'x' + str(int(i))
    cur_y_name = 'y' + str(int(i))
    column_names.append(cur_x_name)
    column_names.append(cur_y_name)
    raw_data.update({cur_x_name: vals[:, 2*i]})
    raw_data.update({cur_y_name: vals[:, 2*i+1]})
#update centers
c_x_name = 'center_x'
c_y_name = 'center_y'
column_names.append(c_x_name)
column_names.append(c_y_name)
raw_data.update({c_x_name: vals[:, 136]})
raw_data.update({c_y_name: vals[:, 137]})

scale_name = 'scale'
column_names.append(scale_name)
raw_data.update({scale_name: vals[:, 138]})

df = pd.DataFrame(raw_data, columns=column_names)
df.to_csv(outname, index=False)

#-----------------------------------------------Loading and Visualization--------------------
# face_dataset = FaceLandmarksDataset(csv_file=outname,
#                                     root_dir=os.path.join(opt.dataroot, 'trainB'))
#
# fig = plt.figure()
#
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#
#     print(i, sample['image'].shape, sample['landmarks'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)
#
#     if i == 3:
#         plt.show()
#         break
print("Finish!")









# preds = preds_v[0,:,:].data.cpu().numpy()
# preds = np.rint(preds).astype(np.int32)
#
# input = io.imread('/home/zeyuan/Desktop/testImages2/train/epoch009_fake_B.png')
# input = cv2.resize(input, dsize=(int(opt.imageSize), int(opt.imageSize)),
#                         interpolation=cv2.INTER_LINEAR)
#
# #TODO: Make this nice
# fig=plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# cv2.polylines(input, [preds[0:17]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(input, [preds[17:22]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(input, [preds[22:27]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(input, [preds[27:31]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(input, [preds[31:36]], 0, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(input, [preds[36:42]], 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(input, [preds[42:48]], 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(input, [preds[48:60]], 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.polylines(input, [preds[60:68]], 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
#
# ax.imshow(input)
# plt.show()
# x = 1
