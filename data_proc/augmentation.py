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

"""Transforms used for augmentation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from collections import Iterable

import cv2
import numpy as np
import paddle
from PIL import Image as PILImage


class Identity(object):
    """identity convert
    """

    def __call__(self, img):
        return img


class RandomTransforms(object):
    """Base class for a list of transformations with randomness

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToPaddleTensor(object):
    """numpy to paddle tensor
    """

    def __call__(self, img):
        return paddle.to_tensor(img.numpy())


class ToArray(object):
    """Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a 
    float array of shape (C x H x W) in the range [0.0, 1.0] 
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or 
    if the numpy.ndarray has dtype = np.uint8
    """

    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        return img.astype('float32')


class Div255(object):
    """Converts a PIL Image or numpy.ndarray in the range [0, 255] to a 
    float array in the range [0.0, 1.0] 
    """

    def __init__(self, return_pil_img=True):
        self.return_pil_img = return_pil_img

    def __call__(self, img):
        img = np.array(img)
        img = img / 255.
        if self.return_pil_img:
            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(img)
        return img


class HWC2CHW(object):
    """Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a 
    float array of shape (C x H x W) in the range [0.0, 1.0] 
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or 
    if the numpy.ndarray has dtype = np.uint8
    """

    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        return img.astype('float32')


class CVResize(object):
    """Resize CV2
    Args:
        size: targe image size
        interpolation: interpolate mode
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return_pil = False
        if isinstance(img, PILImage.Image):
            img = np.array(img)
            return_pil = True

        img = cv2.resize(img, self.size, interpolation=self.interpolation)

        if return_pil:
            img = PILImage.fromarray(img)
        return img
