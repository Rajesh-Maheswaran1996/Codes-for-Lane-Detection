from glob import glob
import os
from dataset.phoenix import save_dataset_rescaled


BASE_PATH = '/media/mykyta/My Passport/phoenix/drive_sim_road_generation/'
TEST_SPLIT = 0.8
# choose from top or realsense camera
CAMERA = 'top'
DOWNSCALED = True

RGB_PREFIX = 'rgb_downscaled' if DOWNSCALED else 'rgb'


def write_image_lists(camera=CAMERA, rgb_prefix='rgb_downscaled'):
    with open('./train_gt_phoenix_{}.txt'.format(camera), 'w+') as train_file, \
            open('./val_gt_phoenix_{}.txt'.format(camera), 'w+') as val_file:
        for folder in glob(os.path.join(BASE_PATH, '*')):
            print('Adding images from folder {}'.format(folder))
            images_in_set = glob(os.path.join(folder, camera, rgb_prefix, '*'))
            for image in images_in_set[:int(len(images_in_set) * TEST_SPLIT)]:
                print('Adding image to train: {}'.format(image))
                train_file.write(image + os.linesep)
            for image in images_in_set[int(len(images_in_set) * TEST_SPLIT):]:
                print('Adding image to val: {}'.format(image))
                val_file.write(image + os.linesep)


write_image_lists(rgb_prefix='rgb')
save_dataset_rescaled()
write_image_lists(rgb_prefix='rgb_downscaled')
