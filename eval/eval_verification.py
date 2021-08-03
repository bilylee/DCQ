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

"""Evaluation script"""

import argparse
import glob
import json
import os

import cv2

cv2.setNumThreads(0)

os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
from scipy import interpolate

import time
import sys

import numpy as np
import paddle

from natsort import natsorted
from paddle.vision import transforms
from paddle.vision.transforms import Normalize

sys.path.append(f'{os.path.dirname(__file__)}/../')

import backbone as models
from data_proc.augmentation import ToArray
from data_proc.image_dataset import EvalDataset
from dcq import DCQ
from train import AverageMeter, ProgressMeter


def parse_eval_args():
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('ckpt_dir', type=str)
    parser.add_argument('epoch', type=int)

    parser.add_argument('--save-prefix', default='', type=str)
    parser.add_argument('--filelist', default='', type=str)
    parser.add_argument('--label-path', default='', type=str)

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--use-flip', default=1, type=int)

    parser.add_argument('--overwrite', default=1, type=int)
    args = parser.parse_args()
    return args


def load_acc_result(filename):
    metas = []
    with open(filename) as fin:
        for line in fin:
            parts = line.strip().replace(',', '').split(' ')
            metas.append([float(parts[1]), float(parts[3]), float(parts[5])])
    return metas


def compute_features(model, use_flip, batch_size, workers, data_path):
    ccrop = transforms.Compose([
        ToArray(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='CHW'),
    ])
    ref_dataset = EvalDataset(
        os.path.dirname(data_path),
        os.path.basename(data_path),
        ccrop)
    eval_loader = paddle.io.DataLoader(
        ref_dataset,
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=workers)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time])
    outputs, targets = [], []
    end = time.time()
    for i, (images, target) in enumerate(eval_loader):
        targets.extend(target)
        # compute output
        output = model(images, im_k=None, use_flip=use_flip, is_train=False)
        outputs.append(output)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            progress.display(i)
    embeddings = paddle.concat(outputs)
    return embeddings, targets


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)
    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    '''
    Copy from [insightface](https://github.com/deepinsight/insightface)
    :param thresholds:
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param far_target:
    :param nrof_folds:
    :return:
    '''
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
                                                        nrof_folds=nrof_folds, pca=pca)
    return tpr, fpr, accuracy, best_thresholds


def eval_verification_base(args):
    # build model
    save_path = os.path.join(
        args.ckpt_dir,
        '{}_performance_{:04d}.txt'.format(args.save_prefix, args.epoch))
    if not args.overwrite and os.path.exists(save_path):
        print(f'{save_path} already exists, skipping due to overwrite is false')
        return
    # parse ckpt args
    config_files = natsorted(glob.glob('{}/args-*.txt'.format(args.ckpt_dir)))
    with open(config_files[-1]) as config_f:
        config = json.load(config_f)

    arch = config['arch']
    print("=> creating model '{}'".format(arch))
    if arch in models.__dict__.keys():
        backbone = models.__dict__[arch]
    else:
        raise NotImplementedError

    model = DCQ(
        backbone,
        config['feat_dim'], queue_size=1)

    # load checkpoint
    resume = os.path.join(
        args.ckpt_dir,
        'face_checkpoint_{:04d}.pdparams'.format(args.epoch))
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = paddle.load(resume)

    model.set_state_dict(checkpoint)
    model.eval()

    # load labels
    labels = np.load(os.path.join(args.label_path))

    embeddings, ref_labels = compute_features(
        model, args.use_flip, args.batch_size, args.workers, args.filelist)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings.numpy(), labels)
    acc, best_thresh = accuracy.mean(), best_thresholds.mean()
    print('[{}] Acc: {:.3f}, best thresh: {}'.format(args.save_prefix, 100 * acc, best_thresh))

    with open(save_path, 'w') as fout:
        fout.write(f'best thresh: {best_thresh}, Acc: {acc}')


def main():
    args = parse_eval_args()
    if args.save_prefix == '':
        args.save_prefix = 'lfw'
    if args.filelist == '':
        args.filelist = './DCQ_train_test_data/common_test_benchmarks/lfw.filelist'
    if args.label_path == '':
        args.label_path = './DCQ_train_test_data/common_test_benchmarks/lfw_label.npy'

    eval_verification_base(args)


if __name__ == '__main__':
    main()
