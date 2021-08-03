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

"""Dataset for loading single data"""

import json
import os

import cv2 as cv
import numpy as np
from paddle.io import Dataset


def opencv_rgb_loader(path):
    """ Read image using opencv's imread function and returns it in bgr format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        return im
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


class ImageDataset(Dataset):
    def __init__(self, root, anno_file, image_loader=opencv_rgb_loader):
        """
        Dataset for training
        :param root: root directory of the dataset
        :param anno_file: annotation json file
            content in the annotation file:
                [
                    [id1/img1.jpg, id1/img2.jpg],
                    [id2/img1.jpg, id2/img2.jpg, id2/img3.jpg],
                    ...
                    [idN/img1.jpg, idN/img2.jpg, ..., idN/imgM.jpg],
                ]
            images from the same list belong to the same face identity
        :param image_loader: function to load images
        """
        super(ImageDataset, self).__init__()

        self.root = root
        self.image_loader = image_loader

        cache_file = os.path.join(root, anno_file)
        print('use anno file: {}'.format(cache_file))
        with open(cache_file, 'r') as f:
            sequence_list = json.load(f)
        print('has {} ids'.format(len(sequence_list)))
        self.sequence_list = sequence_list

        # build index to seq id list
        item_list = []
        for seq_id, seq in enumerate(self.sequence_list):
            for frame_id in range(len(seq)):
                item_list.append((seq_id, frame_id))
        self.item_list = item_list
        print('has {} images'.format(len(item_list)))

    def __len__(self):
        return self.get_num_sequences()

    def __getitem__(self, index):
        """This function won't be used. Use get_frames instead."""
        return None

    def get_num_images(self):
        return len(self.item_list)

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        heads = set([p.split('/')[0] for p in self.sequence_list[seq_id]])
        if len(heads) == 1:
            dataset_name = list(heads)[0]
        else:
            raise NotImplementedError
        return {'seq_len': len(self.sequence_list[seq_id]),
                'set_id': seq_id,
                'dataset': dataset_name}

    def _get_frame(self, sequence, frame_id):
        frame_path = os.path.join(self.root, sequence[frame_id])
        return self.image_loader(frame_path)

    def get_frames(self, seq_id, frame_ids):
        sequence = self.sequence_list[seq_id]
        frame_list = [self._get_frame(sequence, f) for f in frame_ids]
        return frame_list


class EvalDataset(Dataset):
    def __init__(self, root, list_file, transform=None):
        """Dataset for evaluation"""
        super(EvalDataset, self).__init__()
        self.root = root
        with open(os.path.join(root, list_file)) as fin:
            img_list = [p.strip().split(' ') for p in fin]
        self.img_list = img_list
        self.transfrom = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        meta = self.img_list[index]
        img_path = meta[0]
        # dummy label
        label = -1
        label = np.array([label])
        im = cv.imread(os.path.join(self.root, img_path), cv.IMREAD_COLOR)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        if self.transfrom is not None:
            im = self.transfrom(im)
        return im, label
