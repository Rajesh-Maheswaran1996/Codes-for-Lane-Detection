from glob import glob
import os


BASE_PATH = '/media/mykyta/My Passport/phoenix/drive_sim_road_generation/'
TEST_SPLIT = 0.8
# choose from top or realsense camera
CAMERA = 'top'
DOWNSCALED = True

RGB_PREFIX = 'rgb_downscaled' if DOWNSCALED else 'rgb'

with open('./train_gt_phoenix_{}.txt'.format(CAMERA), 'w+') as train_file, \
        open('./val_gt_phoenix_{}.txt'.format(CAMERA), 'w+') as val_file:
    for folder in glob(os.path.join(BASE_PATH, '*')):
        print('Adding images from folder {}'.format(folder))
        images_in_set = glob(os.path.join(folder, CAMERA, RGB_PREFIX, '*'))
        for image in images_in_set[:int(len(images_in_set) * TEST_SPLIT)]:
            print('Adding image to train: {}'.format(image))
            train_file.write(image + os.linesep)
        for image in images_in_set[int(len(images_in_set) * TEST_SPLIT):]:
            print('Adding image to val: {}'.format(image))
            val_file.write(image + os.linesep)
