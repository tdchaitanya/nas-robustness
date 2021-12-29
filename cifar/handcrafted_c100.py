import _init_paths
import os
import numpy as np
from tqdm import tqdm
from functools import partial
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import fastai
import pandas as pd
from fastai.vision.data import ImageDataBunch
from fastai.vision import cifar_stats, models
from fastai.vision.learner import cnn_learner
from fastai.metrics import accuracy
from utils.net_utils import test
from utils.attacks_cifar import pgd_attack_cifar, apgd_attack_cifar


model_names = ['resnet18', 'resnet50', 'densenet121', 'densenet169', 'vgg16_bn']
parser = argparse.ArgumentParser(description='Adversarial attacks on CIFAR-100')
parser.add_argument('--data', default ='../data2',
                    help='path to dataset')
parser.add_argument('--to-csv', default=None,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='batch size to use (on test set)')
parser.add_argument('--eps', default=8/255, type=float,
                    help='Epsilon value for attack, strenght of perturbation')
parser.add_argument('--iters', default=10, type=int,
                    help='number of attack iterations')
parser.add_argument('--alpha', default=2/255, type=float,
                    help='number of attack iterations')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-c', '--checkpoint_dir', default='./c100-weights', type=str,
                    help='path to pre-trained model weights')
parser.add_argument('--attacks', default='all', type=str,
                    help='specify which attack to run, default: all')
parser.add_argument('--runs', default=3, type=int,
                    help='specify number of runs for PGD/AutoPGD, default: 3')

args = parser.parse_args()

print(f'Running adversarial attacks on {args.arch}')
if args.gpu is not None:
    print(f'Using GPU: {args.gpu}')
    torch.cuda.set_device(args.gpu)
device = 'cuda:' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# this part of code is just for invoking the fastai learner
# we'll test on the standard test set only (as used it other cases)
data_path = Path(args.data)
path_c100 = data_path/'cifar100'
train  = path_c100/'train'
data = ImageDataBunch.from_folder(path_c100, train=train, valid_pct=0.2,
        ds_tfms=fastai.vision.transform.get_transforms(), size=32, num_workers=4, bs=4).normalize(cifar_stats)

model = models.__dict__[args.arch]
model = cnn_learner(data, model, metrics=accuracy)
wts_path = os.path.join(args.checkpoint_dir, f'{args.arch}.pth')
net = model.model
net.load_state_dict(torch.load(wts_path)['model'])

if args.gpu is not None:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
if args.to_csv:
    try:
        df = pd.read_csv(args.to_csv)
    except FileNotFoundError:
        df = pd.DataFrame(
                columns=['Run',
                         'Dataset', 
                         'epsilon', 
                         'alpha', 
                         'iterations', 
                         'Network', 
                         'Clean Accuracy', 
                         'PGD', 
                         'PGD_5',
                         'PGD_10',
                         'PGD_20',
                         'AutoPGD',
                         'AutoPGD_5',
                         'AutoPGD_10',
                         'AutoPGD_20',
                        ])

apgd = apgd_attack_cifar(net, testloader, args.eps, args.iters, 10)
print("Average acc on CIFAR100 with AutoPGD-10 on {}: {}".format(args.arch, apgd))  

# pgd_accs = {}
# apgd_accs = {}

# for run in range(args.runs):
#     acc = test(net, testloader)

#     for n_repeats in [1,5,10,20]:
#         pgd_multiple = []
#         for step in range(n_repeats):
#             pgd = pgd_attack_cifar(net, testloader, args.eps, args.alpha, args.iters)
#             pgd_multiple.append(pgd)
#         pgd_accs[n_repeats] = np.average(np.array(pgd_multiple))
#         print("Run {} Average acc on {} starts of PGD on {}: {}".format(run, n_repeats, args.arch, pgd_accs[n_repeats]))

#         apgd = apgd_attack_cifar(net, testloader, args.eps, args.iters, n_repeats)
#         apgd_accs[n_repeats] = apgd
#         print("Run {} Average acc on {} starts of AutoPGD on {}: {}".format(run, n_repeats, args.arch, apgd))    

#     vals = pd.Series({
#             'Run': run,
#             'Dataset': 'cifar100',
#             'epsilon': args.eps,
#             'alpha':args.alpha,
#             'iterations':args.iters,
#             'Network': args.arch, 
#             'Clean Accuracy': acc, 
#             'PGD': pgd_accs[1], 
#             'PGD_5': pgd_accs[5], 
#             'PGD_10': pgd_accs[10], 
#             'PGD_20': pgd_accs[20], 
#             'AutoPGD': apgd_accs[1],
#             'AutoPGD_5': apgd_accs[5],
#             'AutoPGD_10': apgd_accs[10],
#             'AutoPGD_20': apgd_accs[20],
#     }) 

#     df = df.append(vals, ignore_index=True)

# df.to_csv(args.arch + '_cifar100_pgd.csv', index=False)
