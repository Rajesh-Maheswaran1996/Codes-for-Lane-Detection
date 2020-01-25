import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import math
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def radial_gradient(shape, center = None, outer = 200, inner = 100, c1 = np.array([255,255,255]), c2=np.array([0,0,0])):
    i = np.indices(shape)
    z = np.dstack((i[0],i[1]))
    if not center:
        center = np.array([shape[0]/2, shape[1]/2])
    d = np.linalg.norm(z-center, axis=2)
    d = np.minimum(np.maximum((d - inner)/(outer-inner), 0), 1)
    d = np.tile(d.reshape((shape[0], shape[1], 1)), (1,1,3))

    d = d*(c2-c1)+c1
    d=d.astype('uint8')
    return d


class PhoenixDataSet(Dataset):
    """Dataloader for our artifical road segmentation dataset
    """
    def __init__(self, data_list, transform=None, seg_mode='lane_segmentation', visualize=True, radial_mask=True,
                 eval=False, reshape_size=250):
        super(PhoenixDataSet, self).__init__()
        self.seg_mode = seg_mode

        self.seg_folder = 'semseg_color' if seg_mode == 'default' else 'lane_segmentation'

        self.seg_classes = len(all_classes) if seg_mode == 'default' else len(lane_classes)

        self.input_images = []
        with open(os.path.join('list', data_list + '.txt')) as data_file:
            self.input_images = data_file.read().splitlines()

        self.eval = eval
        if not self.eval:
            self.seg_images = [image.replace('rgb', self.seg_folder) for image in self.input_images]

        self.check_existance()

        self.visualize = visualize
        self.radial_mask = radial_mask
        self.transform = transform

        self.classes = all_classes if self.seg_mode == 'default' else lane_classes
        self.height_crop = None

        self.reshape_size = reshape_size
        self.do_center_crop = True

    def check_existance(self):
        images_not_existing = []
        for image in self.input_images:
            if not os.path.exists(image):
                images_not_existing.append(image)

        if not self.eval:
            for image in self.seg_images:
                if not os.path.exists(image):
                    images_not_existing.append(image)

        print('Removing non-existing images: \n {}'.format(images_not_existing))
        for image in images_not_existing:
            if 'rgb' in image:
                try:
                    self.input_images.remove(image)
                except ValueError:
                    pass
                try:
                    self.seg_images.remove(image.replace('rgb', 'semseg_color'))
                except ValueError:
                    pass
            else:
                try:
                    self.seg_images.remove(image)
                except ValueError:
                    pass
                try:
                    self.input_images.remove(image.replace('semseg_color', 'rgb'))
                except ValueError:
                    pass

    def __getitem__(self, idx):
        img = cv2.imread(self.input_images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H = img.shape[0]
        W = img.shape[1]
        if self.height_crop is not None:
            H = int(img.shape[0] * self.height_crop)
            img = img[H:, :, :]

        LOGGER.warn('Loading image: {} and segmentation image: {}'.format(self.input_images[idx],
                    self.input_images[idx].replace('rgb', self.seg_folder)))

        if self.eval:
            seg_img = None
        else:
            print('Loading segmentation image {}'.format(self.input_images[idx].replace('rgb', self.seg_folder)))
            seg_img = cv2.imread(self.input_images[idx].replace('rgb', self.seg_folder), cv2.IMREAD_UNCHANGED)
            trans_mask = seg_img[:, :, 3] == 0
            seg_img[trans_mask] = [0, 0, 0, 255]
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGRA2RGB)
            if self.height_crop is not None:
                seg_img = seg_img[H:, :, :]

        if self.radial_mask:
            scale = (img.shape[0]/512.)

            r = (512/2 - 50)*scale
            outer = r
            inner = r - 15*scale

            d1 = radial_gradient((img.shape[0], img.shape[1]), outer=outer, inner=inner)
            d2 = radial_gradient((img.shape[0], img.shape[1]), outer=inner+1, inner=inner)

            if self.visualize:
                screen_original = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow('Image original', screen_original)

            img = (img.astype('float')*(d1.astype('float')/255.)).astype('uint8')
            if not self.eval:
                seg_img = (seg_img.astype('float')*(d2.astype('float')/255.)).astype('uint8')

            if self.visualize:
                screen_mask = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow('Image with mask', screen_mask)
                cv2.waitKey(0)

        if self.do_center_crop:
            if self.radial_mask:
                # self.center_crop = int(math.sqrt((2 * outer) ** 2 / 2))
                self.center_crop = 1210
            else:
                self.center_crop = 500

            H_center = int(H / 2)
            W_center = int(W / 2)
            seg_img = seg_img[H_center - int(self.center_crop/2): H_center + int(self.center_crop/2),
                              W_center - int(self.center_crop/2): W_center + int(self.center_crop/2), :]
            img = img[H_center - int(self.center_crop/2): H_center + int(self.center_crop/2),
                      W_center - int(self.center_crop/2): W_center + int(self.center_crop/2), :]
            H = img.shape[0]
            W = img.shape[1]

        if self.reshape_size is not None:
            img = cv2.resize(img, (self.reshape_size, self.reshape_size))
            seg_img = cv2.resize(seg_img, (self.reshape_size, self.reshape_size))
            H = self.reshape_size
            W = self.reshape_size

        # 0 -> first class is zero class
        seg_map = np.zeros((H, W), dtype=np.int)

        if not self.eval:
            for ix, color in enumerate(self.classes):
                r = np.all((seg_img[:, :] == color), axis=2)
                seg_map[r] = ix

        if self.visualize:
            if not self.eval:
                screen_seg = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
                cv2.imshow('Segmentation Map', screen_seg)

            screen = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('RGB Image', screen)
            cv2.waitKey(0)

        if self.transform:
            img, seg_map = self.transform((img, seg_map))

        # no right outer lane
        return torch.from_numpy(img).permute(2, 0, 1).contiguous().float(), \
               torch.from_numpy(seg_map).contiguous().long(), np.array([1, 1, 1, 0]), self.input_images[idx]

    def __len__(self):
        return len(self.input_images)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
            exist = torch.stack([b['exist'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]
            exist = [b['exist'] for b in batch]

        samples = {'img': img,
                   'segLabel': segLabel,
                   'exist': exist,
                   'img_name': [x['img_name'] for x in batch]}

        return samples

    @staticmethod
    def get_colors(seg_mode):
        return all_classes if seg_mode == 'default' else lane_classes

    @staticmethod
    def get_weights(seg_mode):
        return [1.]*len(all_classes) if seg_mode == 'default' else [0.4,1,1,1]


LANE_MARKING_SEGMENTATION_COLOR = (128, 0, 0)
BLOCKED_AREA_SEGMENTATION_COLOR = (0, 128, 0)
# pedestrian island currently not segmented, could do in the future
# PEDESTRIAN_ISLAND_COLOR = (0, 0, 128)
DRIVABLE_AREA_SEGMENTATION_COLOR = (0, 255, 0)
STOPLINE_SEGMENTATION_COLOR = (0, 255, 255)
STOPLINE_DASHED_SEGMENTATION_COLOR = (255, 255, 0)
ZEBRA_COLOR = (128, 128, 0)

BACKGROUND_COLOR = (0, 0, 0)

EGO_VEHICLE_COLOR = (100, 100, 100)

OBSTACLE_COLOR = (0, 0, 255)

RAMP_COLOR = (0, 100, 0)

# Traffic markings on the road
# generated by :
# TRAFFIC_MARKING_SEGMENTATION_COLORS = {marking: (255 - 8 * i, 8 * i, 8 * i) for i, marking in
#                                        enumerate(ROADMARKING_TYPE_TO_VISUAL.keys())}
TRAFFIC_MARKING_SEGMENTATION_COLORS = \
    {'10_zone_beginn': (255, 0, 0),
     '20_zone_beginn': (247, 8, 8),
     'stvo-274.1': (239, 16, 16),
     '40_zone_beginn': (231, 24, 24),
     '50_zone_beginn': (223, 32, 32),
     '60_zone_beginn': (215, 40, 40),
     '70_zone_beginn': (207, 48, 48),
     '80_zone_beginn': (199, 56, 56),
     '90_zone_beginn': (191, 64, 64),
     'ende_10_zone': (183, 72, 72),
     'ende_20_zone': (175, 80, 80),
     'stvo-274.2': (167, 88, 88),
     'ende_40_zone': (159, 96, 96),
     'ende_50_zone': (151, 104, 104),
     'ende_60_zone': (143, 112, 112),
     'ende_70_zone': (135, 120, 120),
     'ende_80_zone': (127, 128, 128),
     'ende_90_zone': (119, 136, 136),
     'turn_left': (111, 144, 144),
     'turn_right': (103, 152, 152)}

SIGN_BASE_COLOR = (200, 100, 0)

# Traffjcs signs
# generated by
# SIGN_TO_COLOR = {marking: (7 * i, 255 - 7 * i, 7 * i) for i, marking in
#                  enumerate(SIGN_MESHES.keys())}
SIGN_TO_COLOR = {'10_zone_beginn': (0, 255, 0),
 '20_zone_beginn': (7, 248, 7),
 '40_zone_beginn': (14, 241, 14),
 '50_zone_beginn': (21, 234, 21),
 '60_zone_beginn': (28, 227, 28),
 '70_zone_beginn': (35, 220, 35),
 '80_zone_beginn': (42, 213, 42),
 '90_zone_beginn': (49, 206, 49),
 'ende_10_zone': (56, 199, 56),
 'ende_20_zone': (63, 192, 63),
 'ende_40_zone': (70, 185, 70),
 'ende_50_zone': (77, 178, 77),
 'ende_60_zone': (84, 171, 84),
 'ende_70_zone': (91, 164, 91),
 'ende_80_zone': (98, 157, 98),
 'ende_90_zone': (105, 150, 105),
 'stvo-108-10': (112, 143, 112),
 'stvo-110-10': (119, 136, 119),
 'stvo-205': (126, 129, 126),
 'stvo-206': (133, 122, 133),
 'stvo-208': (140, 115, 140),
 'stvo-209-10': (147, 108, 147),
 'stvo-209-20': (154, 101, 154),
 'stvo-222': (161, 94, 161),
 'stvo-274.1': (168, 87, 168),
 'stvo-274.2': (175, 80, 175),
 'stvo-306': (182, 73, 182),
 'stvo-350-10': (189, 66, 189),
 'stvo-625-10': (196, 59, 196),
 'stvo-625-11': (203, 52, 203),
 'stvo-625-20': (210, 45, 210),
 'stvo-625-21': (217, 38, 217)}

INTERSECTION_COLOR = (64, 128, 255)

# only used in separate lane segmentation map
LANE_MARKING_RIGHT_SIDE = (51, 51, 51)
LANE_MARKING_MIDDLE = (151, 151, 151)
LANE_MARKING_LEFT_SIDE = (251, 251, 251)


def convert_to_one_range(color):
    return (color[0]/255, color[1]/255, color[2]/255)


lane_classes = [BACKGROUND_COLOR, LANE_MARKING_LEFT_SIDE, LANE_MARKING_MIDDLE, LANE_MARKING_RIGHT_SIDE]

all_classes = [
    BACKGROUND_COLOR,
    LANE_MARKING_SEGMENTATION_COLOR,
    BLOCKED_AREA_SEGMENTATION_COLOR,
    DRIVABLE_AREA_SEGMENTATION_COLOR,
    STOPLINE_SEGMENTATION_COLOR,
    STOPLINE_DASHED_SEGMENTATION_COLOR,
    ZEBRA_COLOR,
    EGO_VEHICLE_COLOR,
    OBSTACLE_COLOR,
    RAMP_COLOR
]

class_names = [
    'BACKGROUND_COLOR',
    'LANE_MARKING_SEGMENTATION_COLOR',
    'BLOCKED_AREA_SEGMENTATION_COLOR',
    'DRIVABLE_AREA_SEGMENTATION_COLOR',
    'STOPLINE_SEGMENTATION_COLOR',
    'STOPLINE_DASHED_SEGMENTATION_COLOR',
    'ZEBRA_COLOR',
    'EGO_VEHICLE_COLOR',
    'OBSTACLE_COLOR',
    'RAMP_COLOR'
]
class_names.extend(TRAFFIC_MARKING_SEGMENTATION_COLORS.keys())
class_names.append('SIGN_BASE_COLOR')
class_names.extend(SIGN_TO_COLOR.keys())
class_names.append('INTERSECTION_COLOR')

all_classes.extend(TRAFFIC_MARKING_SEGMENTATION_COLORS.values())
all_classes.append(SIGN_BASE_COLOR)
all_classes.extend(SIGN_TO_COLOR.values())
all_classes.append(INTERSECTION_COLOR)

if __name__ == '__main__':
    # pre-create the scaled-down images
    resave_loader = PhoenixDataSet('train_gt_phoenix_top', visualize=False, radial_mask=True, eval=False,
                                   reshape_size=250)
    for input, target, target_exist, image_name in tqdm(resave_loader):
        new_image_name = image_name.replace('rgb', 'rgb_downscaled')
        if not os.path.exists(os.path.dirname(new_image_name)):
            os.mkdir(os.path.dirname(new_image_name))
            print('Saving rescaled images to: ', os.path.dirname(new_image_name))
            os.mkdir(os.path.dirname(image_name.replace('rgb', 'lane_segmentation_downscaled')))
            print('Saving rescaled segmentation to: ', os.path.dirname(
                image_name.replace('rgb', 'lane_segmentation_downscaled')))

        input = input.permute((1, 2, 0)).numpy()
        input = cv2.cvtColor(input, cv2.COLOR_RGB2BGRA)
        cv2.imwrite(new_image_name, input)
        target = target.numpy()
        seg_map = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.int)
        for ix, color in enumerate(resave_loader.classes):
            r = (target == ix)
            seg_map[r] = color
        cv2.imwrite(image_name.replace('rgb', 'lane_segmentation_downscaled'), seg_map)
