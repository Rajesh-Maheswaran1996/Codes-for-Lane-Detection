import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
import dataset as ds
from options.options import parser
import torch.nn.functional as F

best_mIoU = 0
PROB_THRESHOLD = 0.5


def main():
    global args, best_mIoU
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37
        ignore_label = 255
    elif args.dataset == 'CULane':
        num_class = 5
        ignore_label = 255
    elif args.dataset == 'Phoenix' or 'RealPhoenix':
        num_class = 4  # 3 lanes and ignore label
        ignore_label = 0
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = models.ERFNet(num_class)
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code
    test_dataset = getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(
        args.val_list, visualize=True)
    print(test_dataset.mean, ' std shape ', test_dataset.std)
    # removed random crop as we have a fixed crop area for testing
    test_transform = torchvision.transforms.Compose([
        # tf.GroupRandomCropRatio(size=(args.img_width, args.img_height)),
        tf.GroupNormalize(mean=(test_dataset.mean, (0, )),
                          std=(test_dataset.std, (1, ))),
    ])
    # test_dataset.set_transform(test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(num_class)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(
        ignore_index=ignore_label, weight=class_weights).cuda()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    evaluator = EvalSegmentation(num_class, ignore_label)

    ### evaluate ###
    eval_args = {
        'loader': test_loader,
        'model': model,
        'criterion': criterion,
        'iter': 0,
        'evaluator': evaluator,
        'num_class': num_class - 1,
        'batch_size': args.batch_size}
    if args.dataset == 'RealPhoenix':
        test(**eval_args)
    else:
        validate(**eval_args)
    return


def validate(loader, model, criterion, iter, evaluator, logger=None, num_class=4, batch_size=12):
    batch_time = AverageMeter()
    image_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, _, img_name) in enumerate(loader):

        with torch.no_grad():
            input_var = torch.autograd.Variable(input)

            # compute output
            output, output_exist = model(input_var)

            # measure accuracy and record loss
            output = F.softmax(output, dim=1)

        pred = output.data.cpu().numpy()  # BxCxHxW
        pred_exist = output_exist.data.cpu().numpy()  # BxO

        for cnt in range(len(img_name)):
            cv2.imwrite(os.path.join('predicts', loader.dataset.name,
                                     os.path.basename(img_name[cnt]).rsplit('.')[0] + '_input.png'), input[cnt].numpy().transpose((1, 2, 0)))
            directory = os.path.join('predicts', loader.dataset.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_exist = open(os.path.join('predicts', loader.dataset.name,
                                           os.path.basename(img_name[cnt]).rsplit('.')[0] + '.exist.txt'), 'w')
            for num in range(num_class):
                prob_map = (pred[cnt][num+1]*255).astype(int)
                save_img = cv2.blur(prob_map, (9, 9))
                cv2.imwrite(os.path.join('predicts', loader.dataset.name,
                                         os.path.basename(img_name[cnt]).rsplit('.')[0] + str(num+1) + '_avg.png'), save_img)
                if pred_exist[cnt][num] > PROB_THRESHOLD:
                    file_exist.write('1 ')
                else:
                    file_exist.write('0 ')
            file_exist.close()

            overall_img = pred[cnt]
            overall_img[overall_img < PROB_THRESHOLD] = 0
            overall_img = np.argmax(overall_img, axis=0)
            colored_img = np.zeros(
                (overall_img.shape[0], overall_img.shape[1], 3))
            for num in range(num_class):
                colored_img[overall_img == num] = loader.dataset.classes[num]
            cv2.imwrite(os.path.join('predicts', loader.dataset.name,
                                     os.path.basename(img_name[cnt]).rsplit('.')[0] + '_pred.png'), colored_img)

        # measure elapsed time
        final_time = time.time()
        batch_time.update((final_time - end))
        image_time.update((final_time - end) / batch_size)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                   ' Per image:({image_time.val:.3f}) ({image_time.val:.3f})\t'.format(i, len(loader),
                                                                                       batch_time=batch_time,
                                                                                       image_time=image_time)))

    print('finished, #test:{}'.format(i))

    return mIoU


def test(loader, model, criterion, iter, evaluator, logger=None, num_class=4, batch_size=12):
    batch_time = AverageMeter()
    image_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (test_image, img_name) in enumerate(loader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(test_image)

            # compute output
            output, output_exist = model(input_var)

            # measure accuracy and record loss
            output = F.softmax(output, dim=1)

        pred = output.data.cpu().numpy()  # BxCxHxW
        pred_exist = output_exist.data.cpu().numpy()  # BxO

        for cnt in range(len(img_name)):
            cv2.imwrite(os.path.join('predicts', loader.dataset.name,
                                     img_name[cnt].rsplit('.')[0] + '_input.png'), test_image[cnt].numpy().transpose((1, 2, 0)))
            directory = os.path.join('predicts', loader.dataset.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_exist = open(os.path.join('predicts', loader.dataset.name,
                                           img_name[cnt].rsplit('.')[0] + '.exist.txt'), 'w')
            for num in range(num_class):
                prob_map = (pred[cnt][num+1]*255).astype(int)
                save_img = cv2.blur(prob_map, (9, 9))
                cv2.imwrite(os.path.join('predicts', loader.dataset.name,
                                         img_name[cnt].rsplit('.')[0] + str(num+1) + '_avg.png'), save_img)
                if pred_exist[cnt][num] > 0.5:
                    file_exist.write('1 ')
                else:
                    file_exist.write('0 ')
            file_exist.close()

            overall_img = pred[cnt]
            overall_img = np.argmax(overall_img, axis=0)
            colored_img = np.zeros(
                (overall_img.shape[0], overall_img.shape[1], 3))
            for num in range(num_class):
                colored_img[overall_img == num] = loader.dataset.classes[num]
            cv2.imwrite(os.path.join('predicts', loader.dataset.name,
                                     img_name[cnt].rsplit('.')[0] + '_pred.png'), colored_img)
        # measure elapsed time
        final_time = time.time()
        batch_time.update((final_time - end))
        image_time.update((final_time - end) / batch_size)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                   ' Per image:({image_time.val:.3f}) ({image_time.val:.3f})\t'.format(i, len(loader),
                                                                                       batch_time=batch_time,
                                                                                       image_time=image_time)))

    print('finished, #test:{}'.format(i))

    return mIoU


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class **
                         2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
