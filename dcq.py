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

"""
Dynamic Class Queue (DCQ)
Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dynamic_Class_Queue_for_Large_Scale_Face_Recognition_in_the_CVPR_2021_paper.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
from paddle.nn.functional import normalize

__all__ = ['DCQ']


@paddle.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    if paddle.distributed.get_world_size() < 2:
        return tensor
    tensors_gather = []
    paddle.distributed.all_gather(tensors_gather, tensor)

    output = paddle.concat(tensors_gather, axis=0)
    return output


class DCQ(fluid.dygraph.Layer):
    def __init__(self, base_encoder, dim=128, queue_size=65536,
                 momentum=0.999, scale=50, margin=0.3):
        super(DCQ, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.scale = scale
        self.margin = margin

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, name_prefix='q')
        self.encoder_k = base_encoder(num_classes=dim, name_prefix='k')

        for param_q, param_k in zip(self.encoder_q.parameters(include_sublayers=True),
                                    self.encoder_k.parameters(include_sublayers=True)):
            param_k.stop_gradient = True
            param_q.set_value(param_k)

        self.register_buffer("weight_queue", paddle.randn([dim, queue_size]))
        self.weight_queue = normalize(self.weight_queue, axis=0)

        self.register_buffer("label_queue", paddle.randn([1, queue_size]))
        self.register_buffer("queue_ptr", paddle.zeros([1, ], dtype='int64'))

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            paddle.assign(param_k * self.momentum + param_q * (1. - self.momentum), param_k)
            param_k.stop_gradient = True

    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.weight_queue[:, ptr:ptr + batch_size] = keys.transpose([1, 0])
        self.label_queue[:, ptr:ptr + batch_size] = labels.transpose([1, 0])

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    @paddle.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = paddle.randperm(batch_size_all)

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = paddle.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_shuffle.reshape([num_gpus, -1])[gpu_idx]

        x = paddle.gather(x_gather, idx_this, axis=0)

        return x, idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_unshuffle.reshape([num_gpus, -1])[gpu_idx]

        x = paddle.gather(x_gather, idx_this, axis=0)

        return x

    def forward(self, im_q, im_k=None, im_label=None, use_flip=False, is_train=True):
        if not is_train:
            q = self.encoder_k(im_q)
            if use_flip:
                im_q_flip = paddle.flip(im_q, axis=[3])
                q_flip = self.encoder_k(im_q_flip)
                q = q + q_flip  # no need to divide by 2, which is achieved by normalize
            q = paddle.nn.functional.normalize(q, axis=1)
            return q

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = paddle.nn.functional.normalize(q, axis=1)

        # compute key features
        with paddle.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = paddle.nn.functional.normalize(k, axis=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # positive logits: Nx1
        l_pos = paddle.sum(q * k, axis=1).unsqueeze(-1)
        l_pos = l_pos - self.margin  # apply margin

        # negative logits: NxK
        t_w = self.weight_queue.clone()
        t_w.stop_gradient = True
        l_neg = paddle.matmul(q, t_w)

        # mask out samples with the same label in the queue
        label_diff = im_label - self.label_queue  # N x 1 - 1 x K -> N x K

        mask = (label_diff == 0).astype('float32')
        l_neg = l_neg * (1 - mask) + (-1e9 * mask)

        # logits: Nx(1+K)
        logits = paddle.concat([l_pos, l_neg], axis=1)

        # apply scale
        logits *= self.scale

        # labels: positive key indicators
        labels = paddle.zeros([logits.shape[0], 1], dtype='int64')

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, im_label)

        return logits, labels
