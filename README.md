## Introduction

DCQ (Dynamic Class Queue) is a state-of-the-art face recognition method for training million-IDs datasets.

This repo is the official implementation for CVPR 2021 paper: Dynamic Class Queue for Large Scale Face Recognition In the Wild.
[**[paper]**](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dynamic_Class_Queue_for_Large_Scale_Face_Recognition_in_the_CVPR_2021_paper.pdf)

## News

**`2021-08-03`**: Initial code release.

## Quick Start

### Prerequisite

Install PaddlePaddle 2.1

https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html

Download the MS1MV2 dataset and common test benchmarks via BaiduYun

url: https://pan.baidu.com/s/1PYY3h-jEVURWwQYvLzE3Mg
password: m2m8

```bash
# untar file
cat xaa xab xac xad xae | tar xf -
```

### Training

```bash
# Train iresnet50
bash train_scripts/train_dcq_ir50_q8192_ms1mv2.sh

# Train iresnet100
bash train_scripts/train_dcq_ir100_q8192_ms1mv2.sh
```

### Evaluation

```bash
data_root=./DCQ_train_test_data/common_test_benchmarks
model=Logs/dcq_ires50_q8192_ms1mv2
epoch=19
python eval/eval_verification.py $model $epoch --save-prefix lfw --filelist '$data_root/lfw.filelist' --label-path '$data_root/lfw_label.npy'
python eval/eval_verification.py $model $epoch --save-prefix cplfw --filelist '$data_root/cplfw.filelist' --label-path '$data_root/cplfw_label.npy'
python eval/eval_verification.py $model $epoch --save-prefix agedb_30 --filelist '$data_root/agedb_30.filelist' --label-path '$data_root/agedb_30_label.npy'
```

## Contributing

Main contributors:

- Bi Li
- Jianwei Li
- Nan Peng

## Credit

This code is largely based on [**moco**](https://github.com/facebookresearch/moco) and [**face.evoLVe**](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch).

## Citation

```
@InProceedings{Li_2021_CVPR,
    author    = {Li, Bi and Xi, Teng and Zhang, Gang and Feng, Haocheng and Han, Junyu and Liu, Jingtuo and Ding, Errui and Liu, Wenyu},
    title     = {Dynamic Class Queue for Large Scale Face Recognition in the Wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3763-3772}
}
```