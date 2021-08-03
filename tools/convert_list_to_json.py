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

"""Convert label file to json format"""

import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inp', type=str)
parser.add_argument('out', type=str)
args = parser.parse_args()


def main():
    data_dict = {}
    n_img = 0
    with open(args.inp) as fin:
        for line in fin:
            path, label = line.strip().split(' ')
            n_img += 1
            if label in data_dict:
                data_dict[label].append(path)
            else:
                data_dict[label] = [path]

    all_sequences = []
    for key in data_dict:
        all_sequences.append(data_dict[key])
    print(f'=> num class: {len(all_sequences)}, num img: {n_img}')
    with open(args.out, 'w') as f:
        print('=> saving to {args.out}')
        json.dump(all_sequences, f)


if __name__ == '__main__':
    main()
