# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import builtins
import json
import os
import pickle
import random
import time
import warnings
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader

import backbone as models
from utils.aug_transforms import build_aug
from data_proc.image_dataset import ImageDataset
from data_proc.processing import Processing
from data_proc.sampler import Sampler
from dcq import DCQ

from utils.logger import get_logger

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='DCQ Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('save', metavar='DIR',
                    help='path to save logs and checkpoints')
parser.add_argument('--filelist', default=None, type=str,
                    help='file list used for training')
parser.add_argument('--dataprob', default='1', type=str,
                    help='dataset probability')
parser.add_argument('-a', '--arch', metavar='ARCH', default='iresnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: iresnet50)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--iter-per-epoch', default=11000, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128) in each GPU')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[50, 80, 100], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
# Multi-GPU training related
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--distributed', action='store_true',
                    help='Use multi-processing distributed training')
# DCQ specific configs:
parser.add_argument('--feat-dim', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('--queue-size', default=8192, type=int,
                    help='queue size; number of negative keys (default: 8192)')
parser.add_argument('--dcq-momentum', default=0.999, type=float,
                    help='momentum of updating key encoder (default: 0.999)')
parser.add_argument('--scale', default=50, type=float,
                    help='Cosface loss scale (default: 50)')
parser.add_argument('--margin', default=0.3, type=float,
                    help='Cosface loss margin (default: 0.3)')
# options for data loading and augmentation
parser.add_argument('--sampling-base', default='image', type=str,
                    help='sampling based on category or image')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def main():
    args = parser.parse_args()
    os.makedirs(args.save, exist_ok=True)

    # save the configurations
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    with open(os.path.join(args.save, 'args-{}.txt'.format(timestamp)), 'w') as fh:
        json.dump(args.__dict__, fh, indent=2)

    print('Start at : {}'.format(timestamp))

    # show non-default args
    default_args = parser.parse_args([args.data, args.save])
    for key in args.__dict__:
        if args.__dict__[key] != default_args.__dict__[key]:
            print('{}: {} | default ({})'.format(key, args.__dict__[key],
                                                 default_args.__dict__[key]))

    if args.seed is not None:
        random.seed(args.seed)
        paddle.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = len(paddle.get_cuda_rng_state())
    print('ngpus per node is {}'.format(ngpus_per_node))
    if args.distributed:
        dist.spawn(main_worker, nprocs=ngpus_per_node,
                   args=(args.gpu, ngpus_per_node, args), started_port=6671)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = dist.get_rank()  # None
    logger = get_logger(
        'dcq', log_file='{}/workerlog.{}'.format(args.save, args.gpu),
        level='info', rank=args.gpu)

    # suppress printing if not master
    if args.distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        dist.init_parallel_env()

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    if args.arch in models.__dict__.keys():
        backbone = models.__dict__[args.arch]
    else:
        raise NotImplementedError

    model = DCQ(
        backbone,
        args.feat_dim, args.queue_size, args.dcq_momentum, args.scale, args.margin,
    )

    if args.distributed:
        model = paddle.DataParallel(model)

    criterion = paddle.nn.loss.CrossEntropyLoss(reduction='mean')
    optimizer = paddle.optimizer.Momentum(
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        parameters=model.parameters())

    if args.resume:
        if os.path.isfile(args.resume + '.pdparams'):
            print("=> loading checkpoint '{}'".format(args.resume))
            with open(args.resume + '.state.pickle', 'rb') as fin:
                state = pickle.load(fin)
            args.start_epoch = state['epoch']

            state_dict = paddle.load(args.resume + '.pdparams')
            print(model.set_state_dict(state_dict))

            optimizer_state = paddle.load(args.resume + '.pdopt')
            optimizer.set_state_dict(optimizer_state)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, state['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    augmentation = build_aug(args)

    if args.filelist is not None:
        roots = args.data.split(';')
        anno_files = args.filelist.split(';')
        probs = args.dataprob.split(';')
        assert len(roots) == len(anno_files)
        assert len(probs) == len(anno_files)
        datasets = []
        for root, anno_file in zip(roots, anno_files):
            datasets.append(ImageDataset(root=root, anno_file=anno_file))
        probs = [float(v) for v in probs]
    else:
        raise NotImplementedError

    data_processing = Processing(transform=augmentation)
    train_dataset = Sampler(datasets, probs,
                            samples_per_epoch=args.iter_per_epoch * args.batch_size,
                            processing=data_processing,
                            k=2,
                            sampling_base=args.sampling_base)

    if args.sampling_base == 'image':
        train_sampler = paddle.io.DistributedBatchSampler(
            train_dataset, args.batch_size, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_dataset, num_workers=1, batch_sampler=train_sampler)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True)
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}: DataLoader is ready.')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, logger)
        if args.gpu == 0 and epoch > args.schedule[0]:
            # skip saving the queue
            state_dict = {}
            model_state_dict = model.state_dict()
            for key in model_state_dict:
                # we don't need to save the queue
                if 'queue' not in key:
                    state_dict[key] = model_state_dict[key]

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(), },
                filename='{}/face_checkpoint_{:04d}'.format(args.save, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, logger=None):
    batch_time = MoveAverageMeter('Time', ':.3f')
    data_time = MoveAverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    top5 = AverageMeter('Acc@5', ':3.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}: start training.')
    for i, (image_q, image_k, set_id) in enumerate(train_loader):
        if i >= args.iter_per_epoch:
            break
        images = [image_q, image_k]

        # measure data loading time
        _dt = time.time() - end
        data_time.update(_dt)
        # compute output
        output, target = model(im_q=images[0], im_k=images[1], im_label=set_id)
        _inft = time.time() - end

        loss = criterion(output, target)

        _losst = time.time() - end

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.numpy()[0], images[0].shape[0])
        top1.update(acc1.numpy()[0], images[0].shape[0])
        top5.update(acc5.numpy()[0], images[0].shape[0])

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # measure elapsed time
        _bt = time.time() - end
        batch_time.update(_bt)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, optimizer.get_lr(), logger=logger)


def save_checkpoint(state, filename='checkpoint'):
    paddle.save(state['state_dict'], filename + '.pdparams')
    del state['state_dict']
    paddle.save(state['optimizer'], filename + '.pdopt')
    del state['optimizer']
    with open(filename + '.state.pickle', 'wb') as fout:
        pickle.dump(state, fout)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MoveAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', win=1000):
        self.name = name
        self.fmt = fmt
        self.win = win
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.move_win = []

    def update(self, val):
        self.val = val
        self.move_win.append(val)
        if len(self.move_win) > self.win:
            self.move_win.pop(0)
        self.avg = np.mean(self.move_win)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, append_info=None, logger=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if append_info is not None:
            entries.append('LR {:.1e}'.format(append_info))
        if logger is not None:
            logger.info(' '.join(entries))
        else:
            print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    optimizer.set_lr(lr)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = output.detach()
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = paddle.topk(output, maxk, 1, True, True)
    pred = pred.t()
    correct = pred.equal(paddle.reshape(target, (1, -1)).expand_as(pred)).astype('float32')

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(0, keepdim=True)
        res.append(correct_k * (100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
