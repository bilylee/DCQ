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

"""Logging utilities"""

import os
import logging
from logging import handlers


def get_logger(
        name, log_file=None, level='info', rank=0,
        fmt='%(asctime)s-%(levelname)s: %(message)s'):
    # fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    log_dir = os.path.split(log_file)[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }
    logger = logging.getLogger(name)
    format_str = logging.Formatter(fmt)
    logger.setLevel(level_relations.get(level))
    if rank == 0:
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        logger.addHandler(sh)
    if log_file is not None:
        th = logging.FileHandler(filename=log_file, encoding='utf-8')
        th.setFormatter(format_str)
        logger.addHandler(th)

    return logger
