import _init_paths
import os
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from pcdarts import pcdarts_utils
from pcdarts import genotypes
from pcdarts.model import NetworkCIFAR as Network
import pandas as pd

from utils.net_utils import test
from utils.attacks_cifar import auto_attack_cifar, fgsm_attack_cifar, rfgsm_attack_cifar, pgd_attack_cifar, apgd_attack_cifar

parser = argparse.ArgumentParser(description='Adversarial attacks on CIFAR (PC-DARTS)')
parser.add_argument('--data', default ='../data',
                    help='path to dataset')
parser.add_argument('--to-csv', default=None,
                    help='path to dataset')
parser.add_argument('--cifar100', action='store_true', default=False, help='run attacks on cifar100 dataset')
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
parser.add_argument('-c', '--checkpoint_dir', default='../pcdarts/pretrained/', type=str,
                    help='path to pre-trained model weights')
parser.add_argument('--attacks', default='all', type=str,
                    help='specify which attack to run, default: all')
parser.add_argument('--runs', default=3, type=int,
                    help='specify number of runs for PGD/AutoPGD, default: 3')

args = parser.parse_args()
np.random.seed(0)
torch.manual_seed(0)

if args.gpu is not None:
    print(f'Using GPU: {args.gpu}')
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.manual_seed(0)

if args.cifar100:
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
  ])

  testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
  CLASSES = 100
  path = os.path.join(args.checkpoint_dir, 'cifar100.pt')


else:
  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
  CLASSES = 10
  path = os.path.join(args.checkpoint_dir, 'cifar10.pt')


genotype = eval("genotypes.%s" % 'PCDARTS')
net = Network(36, CLASSES, 20, True, genotype)
net = net.cuda()

pcdarts_utils.load(net, path)
net.drop_path_prob = 0.2

print("PC-DARTS")
apgd = apgd_attack_cifar(net, testloader, args.eps, args.iters, 10)
print("Average acc of AutoPGD-10 on PC-DARTS: {}".format(apgd))
# if args.to_csv:
#     try:
#         df = pd.read_csv(args.to_csv)
#     except FileNotFoundError:
#         df = pd.DataFrame(
#                 columns=['Run',
#                          'Dataset', 
#                          'epsilon', 
#                          'alpha', 
#                          'iterations', 
#                          'Network', 
#                          'Clean Accuracy', 
#                          'PGD', 
#                          'PGD_5',
#                          'PGD_10',
#                          'PGD_20',
#                          'AutoPGD',
#                          'AutoPGD_5',
#                          'AutoPGD_10',
#                          'AutoPGD_20',
#                         ])

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
#         print("Run {} Average acc on {} starts of PGD on {}: {}".format(run, n_repeats, 'pcdarts', pgd_accs[n_repeats]))

#         apgd = apgd_attack_cifar(net, testloader, args.eps, args.iters, n_repeats)
#         apgd_accs[n_repeats] = apgd
#         print("Run {} Average acc on {} starts of AutoPGD on {}: {}".format(run, n_repeats, 'pcdarts', apgd))    

#     vals = pd.Series({
#             'Run': run,
#             'Dataset': 'cifar10',
#             'epsilon': args.eps,
#             'alpha':args.alpha,
#             'iterations':args.iters,
#             'Network': 'pcdarts', 
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

# df.to_csv('pcdarts_cifar10.csv', index=False)
