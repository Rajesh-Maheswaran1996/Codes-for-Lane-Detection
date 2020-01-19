from glob import glob
import os


BASE_PATH = '/home/ubuntu/mounted_synthetic_for_lane_detection'
TEST_SPLIT = 0.8

with open('./train_gt_phoenix.txt', 'a') as train_file, open('./val_gt_phoenix.txt', 'a') as val_file:
    for folder in glob(os.path.join(BASE_PATH, '*')):
        print('Adding images from folder {}'.format(folder))
        images_in_set = glob(os.path.join(folder, 'rgb', '*'))
        for image in images_in_set[:int(len(images_in_set) * TEST_SPLIT)]:
            print('Adding image to train: {}'.format(image))
            train_file.write(image + os.linesep)
        for image in images_in_set[int(len(images_in_set) * TEST_SPLIT):]:
            print('Adding image to val: {}'.format(image))
            val_file.write(image + os.linesep)
