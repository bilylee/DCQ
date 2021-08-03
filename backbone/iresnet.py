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

"""Improved ResNet backbone"""

from collections import namedtuple

import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from paddle.fluid.initializer import Constant
from paddle.framework import get_default_dtype
from paddle.nn import (
    Linear, Conv2D, BatchNorm1D, BatchNorm2D, ReLU,
    Sigmoid, Dropout, MaxPool2D, AdaptiveAvgPool2D, Sequential,
    Layer, Flatten)

__all__ = ['iresnet34', 'iresnet50', 'iresnet100', 'iresnet50_se', 'iresnet100_se']


class PReLU(Layer):
    def __init__(self, num_parameters=1, init=0.25, weight_attr=None,
                 name=None):
        super(PReLU, self).__init__()
        self._num_parameters = num_parameters
        self._init = init
        self._weight_attr = weight_attr
        self._name = name

        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=[self._num_parameters],
            dtype=get_default_dtype(),
            is_bias=False,
            default_initializer=Constant(self._init))

    def forward(self, x):
        return F.prelu(x, self.weight)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'num_parameters={}, init={}, dtype={}{}'.format(
            self._num_parameters, self._init, self._dtype, name_str)


def l2_norm(input, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = paddle.divide(input, norm)
    return output


class SEModule(Layer):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.fc1 = Conv2D(
            channels, channels // reduction, kernel_size=1, padding=0, bias_attr=False)
        self.relu = ReLU()
        self.fc2 = Conv2D(
            channels // reduction, channels, kernel_size=1, padding=0, bias_attr=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(Layer):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2D(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2D(in_channel, depth, (1, 1), stride, bias_attr=False),
                BatchNorm2D(depth))
        self.res_layer = Sequential(
            BatchNorm2D(in_channel),
            Conv2D(in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False), PReLU(depth),
            Conv2D(depth, depth, (3, 3), stride, 1, bias_attr=False), BatchNorm2D(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE(Layer):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2D(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2D(in_channel, depth, (1, 1), stride, bias_attr=False),
                BatchNorm2D(depth))
        self.res_layer = Sequential(
            BatchNorm2D(in_channel),
            Conv2D(in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False),
            PReLU(depth),
            Conv2D(depth, depth, (3, 3), stride, 1, bias_attr=False),
            BatchNorm2D(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise NotImplementedError

    return blocks


class Backbone(Layer):
    def __init__(self, input_size, num_layers, out_dim, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [34, 50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleneckIR
        elif mode == 'ir_se':
            unit_module = BottleneckIRSE
        self.input_layer = Sequential(Conv2D(3, 64, (3, 3), 1, 1, bias_attr=False),
                                      BatchNorm2D(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2D(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, out_dim),
                                           BatchNorm1D(out_dim))
        else:
            self.output_layer = Sequential(BatchNorm2D(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, out_dim),
                                           BatchNorm1D(out_dim))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


def iresnet34(num_classes, input_size=[112, 112], **kwargs):
    model = Backbone(input_size, 34, num_classes, 'ir')
    return model


def iresnet50(num_classes, input_size=[112, 112], **kwargs):
    model = Backbone(input_size, 50, num_classes, 'ir')
    return model


def iresnet100(num_classes, input_size=[112, 112], **kwargs):
    model = Backbone(input_size, 100, num_classes, 'ir')
    return model


def iresnet50_se(num_classes, input_size=[112, 112], **kwargs):
    model = Backbone(input_size, 50, num_classes, 'ir_se')
    return model


def iresnet100_se(num_classes, input_size=[112, 112], **kwargs):
    model = Backbone(input_size, 100, num_classes, 'ir_se')
    return model
