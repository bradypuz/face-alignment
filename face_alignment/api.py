from __future__ import print_function
import numpy as np
import dlib
import math
from enum import Enum
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """Initialize the face alignment pipeline

    Args:
        landmarks_type (``LandmarksType`` object): an enum defining the type of predicted points.
        network_size (``NetworkSize`` object): an enum defining the size of the network (for the 2D and 2.5D points).
        enable_cuda (bool, optional): If True, all the computations will be done on a CUDA-enabled GPU (recommended).
        enable_cudnn (bool, optional): If True, cudnn library will be used in the benchmark mode
        flip_input (bool, optional): Increase the network accuracy by doing a second forward passed with
                                    the flipped version of the image
        use_cnn_face_detector (bool, optional): If True, dlib's CNN based face detector is used even if CUDA
                                                is disabled.

    Example:
        >>> FaceAlignment(NetworkSize.2D, flip_input=False)
    """

    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 enable_cuda=True, enable_cudnn=True, flip_input=False,
                 use_cnn_face_detector=False, gpu_ids=[]):
        self.enable_cuda = enable_cuda
        self.use_cnn_face_detector = use_cnn_face_detector
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.gpu_ids = gpu_ids
        base_path = os.path.join(appdata_dir('face_alignment'), "data")

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if enable_cudnn and self.enable_cuda:
            torch.backends.cudnn.benchmark = True

        # Initialise the face detector
        if self.enable_cuda or self.use_cnn_face_detector:
            path_to_detector = os.path.join(
                base_path, "mmod_human_face_detector.dat")
            if not os.path.isfile(path_to_detector):
                print("Downloading the face detection CNN. Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat",
                    os.path.join(path_to_detector))

            self.face_detector = dlib.cnn_face_detection_model_v1(
                path_to_detector)

        else:
            self.face_detector = dlib.get_frontal_face_detector()

        # Initialise the face alignemnt networks
        self.face_alignemnt_net = FAN(int(network_size))
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(int(network_size)) + '.pth.tar'
        else:
            network_name = '3DFAN-' + str(int(network_size)) + '.pth.tar'
        fan_path = os.path.join(base_path, network_name)

        if not os.path.isfile(fan_path):
            print("Downloading the Face Alignment Network(FAN). Please wait...")

            request_file.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/" +
                network_name, os.path.join(fan_path))

        fan_weights = torch.load(
            fan_path,
            map_location=lambda storage,
            loc: storage)

        self.face_alignemnt_net.load_state_dict(fan_weights)

        if self.enable_cuda:
            self.face_alignemnt_net.cuda(device=self.gpu_ids[0])


        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()
            depth_model_path = os.path.join(base_path, 'depth.pth.tar')
            if not os.path.isfile(depth_model_path):
                print(
                    "Downloading the Face Alignment depth Network (FAN-D). Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/python-fan/depth.pth.tar",
                    os.path.join(depth_model_path))

            depth_weights = torch.load(
                depth_model_path,
                map_location=lambda storage,
                loc: storage)
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            if self.enable_cuda:
                self.depth_prediciton_net.cuda(device=self.gpu_ids[0])
            self.depth_prediciton_net.eval()

    def detect_faces(self, image):
        """Run the dlib face detector over an image

        Args:
            image (``ndarray`` object or string): either the path to the image or an image previosly opened
            on which face detection will be performed.

        Returns:
            Returns a list of detected faces
        """
        return self.face_detector(image, 1)

    def get_landmarks(self, input_image, all_faces=False, use_cuda = True):
        bs, nc, h, w = input_image.size()
        images_numpy = []
        centers = []
        scales = []
        # heatmaps_list = Variable(torch.zeros(bs, 68, 64, 64))       #the default size of FAN output
        inps = Variable(torch.zeros(bs, nc, 256, 256))              #the default size of inputs of FAN is 256
        if use_cuda:
            # heatmaps_list = heatmaps_list.cuda()
            inps = inps.cuda(device=self.gpu_ids[0])
        for i in range(bs):
            image = input_image[i].data.cpu().float().numpy()
            image = (np.transpose(image, (1, 2, 0)) + 1)  * 255.0 / 2.0
            # image = np.transpose(image, (1, 2, 0))  * 255.0
            image = np.rint(image).astype(np.uint8)
            images_numpy.append(image)
            detected_face_list = self.detect_faces(image)

            # print(len(detected_face_list))
            if (len(detected_face_list) == 1):
            # assert (len(detected_face_list) == 1)                        #assuming there is only one face in each input image
                detected_face = detected_face_list[-1]
                bbox = detected_face.rect
                center = torch.FloatTensor(
                    [bbox.right() - (bbox.right() - bbox.left()) / 2.0, bbox.bottom() -
                     (bbox.bottom() - bbox.top()) / 2.0])
                scale = (bbox.right() - bbox.left() + bbox.bottom() - bbox.top()) / 195.0

                center[1] = center[1] - (bbox.bottom() - bbox.top()) * 0.12
            else:
                center = torch.FloatTensor([0.5*h, 0.56*w])
                scale = 1.17 * (h + w) / (195.0 * 2)
            if use_cuda:
                center = center.cuda(device=self.gpu_ids[0])

            inp = crop(input_image[i], center, scale, gpu_ids=self.gpu_ids)
            inp = inp.add(1).div(2)
            centers.append(center)
            scales.append(scale)
            inps[i, :, :, :] = inp[0, :, :, :]
        heatmaps_list = self.face_alignemnt_net(inps)               #only use the last feature maps
        pts, pts_img = get_preds_fromhm(heatmaps_list[-1], centers, scales,gpu_ids=self.gpu_ids)
        pts, pts_img = pts.view(bs, 68, 2) * 4, pts_img.view(bs, 68, 2)

        #generate 3d landmarks

        # if self.landmarks_type == LandmarksType._3D:
        #     for i in range(bs):
        #         heatmaps = np.zeros((68, 256, 256))
        #         for j in range(68):
        #             if pts[i, j, 0].data > 0:
        #                 heatmaps[i] = draw_gaussian(heatmaps[i], pts[i,j], 2)
        #         heatmaps = torch.from_numpy(
        #             heatmaps).view(1, 68, 256, 256).float()
        #         if self.enable_cuda:
        #             heatmaps = heatmaps.cuda()
        #         depth_pred = self.depth_prediciton_net(
        #             Variable(
        #                 torch.cat(
        #                     (inp, heatmaps), 1), volatile=True)).data.cpu().view(68, 1)
        #         pts_img = torch.cat(
        #             (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)


        return pts_img,heatmaps_list, centers, scales


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