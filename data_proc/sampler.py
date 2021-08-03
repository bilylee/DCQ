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

"""Images Samplers"""

import random

import numpy as np
from paddle.io import Dataset


class Sampler(Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, processing,
                 k=2, sampling_base='category', random_idx=False):
        """
        :param datasets: a list of dataset from which to sample images
        :param p_datasets: probabilities of datasets to be sampled
        :param samples_per_epoch: how many samples to form an epoch
        :param processing: data processing function
        :param k: number of images to be sampled from the same face identity
        :param sampling_base: 'image' or 'category'
            for 'image', each image in the dataset will sampled with equal probability
            for 'category', each identity in the dataset will be sampled with equal probability
        :param random_idx: whether to use random identity in the dataset
        """
        super(Sampler, self).__init__()

        self.datasets = datasets
        if p_datasets is None:
            p_datasets = [1.0 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.processing = processing
        self.k = k
        self.sampling_base = sampling_base
        self.random_idx = random_idx

        # the start index for generating labels from different datasets
        self.start_idx_list = np.cumsum([0] + [d.get_num_sequences() for d in datasets])

    def __len__(self):
        if self.sampling_base == 'category':
            return self.samples_per_epoch
        elif self.sampling_base == 'image':
            return self.datasets[0].get_num_images()
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        # Select a dataset
        dataset_idx = random.choices(range(len(self.datasets)), self.p_datasets)[0]
        dataset = self.datasets[dataset_idx]
        start_idx = self.start_idx_list[dataset_idx]

        if self.sampling_base == 'category':
            seq_id = random.sample(range(dataset.get_num_sequences()), k=1)[0]
            seq_info = dataset.get_sequence_info(seq_id)

            if seq_info['seq_len'] == 1:
                frame_ids = [0] * self.k
            else:
                frame_ids = random.choices(range(seq_info['seq_len']), k=self.k)
        elif self.sampling_base == 'image':
            if self.random_idx or len(self.datasets) > 0:
                item_idx = random.sample(range(dataset.get_num_images()), k=1)[0]
            else:
                item_idx = index
            seq_id, query_id = dataset.item_list[item_idx]
            seq_info = dataset.get_sequence_info(seq_id)

            if seq_info['seq_len'] == 1:
                frame_ids = [0] * self.k
            else:
                query_id = [query_id]  # as list
                key_ids = random.choices(range(seq_info['seq_len']), k=self.k - 1)
                frame_ids = query_id + key_ids
        else:
            raise NotImplementedError

        images = dataset.get_frames(seq_id, frame_ids)

        # Prepare data
        data = {'images': images,
                'dataset': seq_info['dataset'],
                'set_id': seq_id + start_idx}

        # Send for processing
        data = self.processing(data)
        return data
