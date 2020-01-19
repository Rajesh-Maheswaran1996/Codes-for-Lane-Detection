import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def radial_gradient(shape, center = None, outer = 200, inner = 100, c1 = np.array([255,255,255]), c2=np.array([0,0,0])):
    i = np.indices(shape)
    z = np.dstack((i[0],i[1]))
    if center == None:
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
    def __init__(self, data_list, transforms=None, seg_mode='lane_segmentation', augment=False, preprocess=False,
                 visualize=False, radial_mask=True, eval=False, transform=None):
        super(PhoenixDataSet, self).__init__()
        self.transforms = transforms
        self.seg_mode = seg_mode

        self.seg_folder = 'semseg_color' if seg_mode == 'default' else 'lane_segmentation'

        self.seg_classes = len(all_classes) if seg_mode == 'default' else len(lane_classes)

        self.input_images = []
        with open(os.path.join('list', data_list + '.txt')) as data_file:
            self.input_images = data_file.read().splitlines()

        self.eval = eval
        if not self.eval:
            self.seg_images = [image.replace('rgb', 'semseg_color') for image in self.input_images]

        self.check_existance()

        self.visualize = visualize
        self.radial_mask = radial_mask
        self.transform = transform

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

        if self.eval:
            seg_img = None
        else:
            seg_img = cv2.imread(self.input_images[idx], cv2.IMREAD_UNCHANGED)
            trans_mask = seg_img[:, :, 3] == 0
            seg_img[trans_mask] = [0, 0, 0, 255]
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGRA2RGB)

        if self.radial_mask:
            scale = (img.shape[0]/512.)

            r = (512/2 - 50)*scale
            outer = r
            inner = r - 15*scale

            d1 = radial_gradient((img.shape[0], img.shape[1]), outer=outer, inner=inner)
            d2 = radial_gradient((img.shape[0], img.shape[1]), outer=inner+1, inner=inner)
            img = (img.astype('float')*(d1.astype('float')/255.)).astype('uint8')
            if not self.eval:
                seg_img = (seg_img.astype('float')*(d2.astype('float')/255.)).astype('uint8')

        if self.visualize:
            if not self.eval:
                screen = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
                cv2.imshow('Test', screen)
                cv2.waitKey()

            screen = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Test', screen)
            cv2.waitKey()

        print('Loading image {}'.format(self.input_images[idx]))

        sample = {
            'img': img,
            'segLabel': seg_img,
            'exist': None,
            'img_name': '{}'.format(self.input_images[idx])
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        W = sample['img'].shape[0]
        H = sample['img'].shape[1]
        # 0 -> first class is zero class
        segMap = np.zeros((W, H), dtype=np.int)

        classes = all_classes if self.seg_mode == 'default' else lane_classes

        if not self.eval:
            for ix, color in enumerate(classes):
                r = np.all((sample['segLabel'][:, :] == color), axis=2)
                segMap[r] = ix

            sample['segLabel'] = segMap

            exist = np.zeros(len(classes), dtype=np.float32)
            for ix in range(len(classes)):
                exist[ix] = (segMap == ix).any()
            sample['exist'] = exist
            sample['segLabel'] = sample['segLabel'].squeeze()
            print('Unique in sample: {}'.format(np.unique(sample['segLabel'])))

        if self.transform:
            sample['img'], sample['segLabel'] = self.transform((sample['img'], sample['segLabel']))

        # no right outer lane
        return torch.from_numpy(sample['img']).permute(2, 0, 1).contiguous().float(), \
               torch.from_numpy(sample['segLabel']).contiguous().long(), np.array([1, 1, 1, 0])

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
