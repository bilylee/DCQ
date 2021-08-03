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

"""utils for data augmentation"""

from paddle.vision.transforms import (Compose, ColorJitter,
                                      Grayscale, RandomHorizontalFlip, RandomCrop, Resize, Normalize)

from data_proc.augmentation import (Identity,
                                    RandomApply, Div255, HWC2CHW, ToArray)


def build_aug(args):
    """build augmentation transforms
    Args:
        args: data augmentation config
    Return: transforms
    """
    transforms = Compose([
        RandomHorizontalFlip(),
        ToArray(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='CHW'),
    ])
    return transforms
