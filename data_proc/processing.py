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

"""Processing images with transforms"""

import numpy as np
from PIL import Image


class Processing:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        new_images = []
        for idx, img in enumerate(data['images']):
            img = Image.fromarray(img)
            transform = self.transform
            new_images.append(transform(img))

        data['images'] = new_images
        data['set_id'] = np.array([float(data['set_id'])], dtype=np.float32)
        return data['images'][0], data['images'][1], data['set_id']
