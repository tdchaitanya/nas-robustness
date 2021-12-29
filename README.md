![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [On Adversarial Robustness: A Neural Architecture Search perspective](https://arxiv.org/abs/2007.08428)


## Preparation:

Clone the repository:
```
https://github.com/tdchaitanya/nas-robustness.git
```

### prerequisites

* Python 3.6
* Pytorch 1.2.0
* CUDA 10.1

For a hassle-free environment setup, use the `environment.yml` file included in the repository.

### Pre-trained models:

For easy reproduction of the result shown in the paper, this repository is organized dataset-wise, and all the pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1jrOuEBQ3lFDbI916ps8lEFPvRyOCZqLl?usp=sharing)

## CIFAR-10/100

All the commands in this section should be executed in the `cifar` directory.

##### Hand-crafted models on CIFAR-10

All the files corresponding to this dataset are included in `cifar-10/100` directories. Download `cifar` weigths from the shared [drive link](https://drive.google.com/drive/folders/1jrOuEBQ3lFDbI916ps8lEFPvRyOCZqLl?usp=sharing) and place them in `nas-robustness/cifar-10/cifar10_models/state_dicts` directory.

For running all the four attacks on `Resnet-50` (shown in Table 1) run the following command.

```
python handcrafted.py --arch resnet50
```

Change the architecture parameter to run attacks on other models. Only resnet-18, resnet-50, densenet-121, densenet-169, vgg-16 are supported for now. For other models, you may have to train them from scratch before running these attacks.

##### Hand-crafted models on CIFAR-100
For training the models on CIFAR-100 we have used `fastai` library. Download `cifar-100` weigths from the shared [drive link](https://drive.google.com/drive/folders/1jrOuEBQ3lFDbI916ps8lEFPvRyOCZqLl?usp=sharing) and place them in `nas-robustness/cifar/c100-weights` directory.

Additionally, you'll also have to download the CIFAR-100 dataset from [here](https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz) and place it in the data directory (we'll not be using this anywhere, this is just needed to initialize the fastai model).

```
python handcrafted_c100.py --arch resnet50
```

##### DARTS

Download DARTS CIFAR-10/100 weights from the [drive](https://drive.google.com/drive/folders/1jrOuEBQ3lFDbI916ps8lEFPvRyOCZqLl?usp=sharing) and place it `nas-robustness/darts/pretrained`

For running all the four attacks on `DARTS` run the following command:

```
python darts-nas.py
```
Add --cifar100 to run the experiments on cifar-100

##### P-DARTS

Download P-DARTS CIFAR-10/100 weights from the [drive](https://drive.google.com/drive/folders/1jrOuEBQ3lFDbI916ps8lEFPvRyOCZqLl?usp=sharing) and place it `nas-robustness/pdarts/pretrained`

For running all the four attacks on `P-DARTS` run the following command:

```
python pdarts-nas.py
```
Add --cifar100 to run the experiments on CIFAR-100

##### NSGA-Net

Download NSGA-Net CIFAR-10/100 weights from the [drive](https://drive.google.com/drive/folders/1jrOuEBQ3lFDbI916ps8lEFPvRyOCZqLl?usp=sharing) and place it `nas-robustness/nsga_net/pretrained`

For running all the four attacks on `P-DARTS` run the following command:

```
python nsganet-nas.py
```
Add --cifar100 to run the experiments on CIFAR-100

##### PC-DARTS

Download PC-DARTS CIFAR-10/100 weights from the [drive](https://drive.google.com/drive/folders/1jrOuEBQ3lFDbI916ps8lEFPvRyOCZqLl?usp=sharing) and place it `nas-robustness/pcdarts/pretrained`

For running all the four attacks on `PC-DARTS` run the following command:

```
python pcdarts-nas.py
```
Add --cifar100 to run the experiments on CIFAR-100

## ImageNet

All the commands in this section should be executed in `ImageNet` directory.

##### Hand-crafted models

All the files corresponding to this dataset are included in `imagenet` directory. We use the default pre-trained weights provided by PyTorch for all attacks.

For running all the four attacks on `Resnet-50`  run the following command:

```
python handcrafted.py --arch resnet50
```

**For DARTS, P-DARTS, PC-DARTS follow the same instructions as mentioned above for CIFAR-10/100, just change the working directory to `ImageNet`**

#### DenseNAS

Download DenseNAS ImageNet weights from the [drive](https://drive.google.com/drive/folders/1jrOuEBQ3lFDbI916ps8lEFPvRyOCZqLl?usp=sharing) (these are same as the weights provided in thier official repo) and place it `nas-robustness/densenas/pretrained`

For running all the four attacks on `DenseNAS-R3` run the following command:

```
python dense-nas.py --model DenseNAS-R3
```

### Citation
```
@InProceedings{Devaguptapu_2021_ICCV,
    author    = {Devaguptapu, Chaitanya and Agarwal, Devansh and Mittal, Gaurav and Gopalani, Pulkit and Balasubramanian, Vineeth N},
    title     = {On Adversarial Robustness: A Neural Architecture Search Perspective},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {152-161}
}
```
### Acknowledgements

Some of the code and weights provided in this library are borrowed from the libraries mentioned below:
- [cifar10_models](https://github.com/huyvnphan/PyTorch-CIFAR10)
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [fastai](https://github.com/fastai/fastai)
- [darts](https://github.com/quark0/darts)
- [pdarts](https://github.com/chenxin061/pdarts)
- [nsga-net](https://github.com/ianwhale/nsga-net)
- [pcdarts](https://github.com/yuhuixu1993/PC-DARTS)
- [densenas](https://github.com/JaminFong/DenseNAS)
