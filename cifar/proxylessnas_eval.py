import _init_paths
import os
import os.path as osp
import numpy as np
import argparse
import time
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets
import torchvision

from proxyless_nas.utils import AverageMeter, accuracy

from proxyless_nas import model_zoo
from utils.net_utils import test
from utils.attacks_cifar import auto_attack_cifar, fgsm_attack_cifar, rfgsm_attack_cifar, pgd_attack_cifar

model_names = sorted(name for name in model_zoo.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(model_zoo.__dict__[name]))

parser = argparse.ArgumentParser(description='Adversarial attacks on CIFAR (Proxyless-NAS)')
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
parser.add_argument('--eps', default=0.007, type=float,
                    help='Epsilon value for attack, strenght of perturbation')
parser.add_argument('--iters', default=3, type=int,
                    help='number of attack iterations')
parser.add_argument('--alpha', default=0.03, type=float,
                    help='number of attack iterations')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')

parser.add_argument('-c', '--checkpoint_dir', default='../pcdarts/pretrained/', type=str,
                    help='path to pre-trained model weights')
parser.add_argument('--attacks', default='all', type=str,
                    help='specify which attack to run, default: all')

net = model_zoo.__dict__['proxyless_cifar'](pretrained=True)

args = parser.parse_args()
np.random.seed(0)
torch.manual_seed(0)

if args.cifar100:
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
  ])

  testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

else:
  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

device = torch.device('cuda:0')

net = torch.nn.DataParallel(net).to(device)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().to(device)

net.eval()

losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

# with torch.no_grad():
#     with tqdm(total=len(data_loader), desc='Test') as t:
#         for i, (_input, target) in enumerate(data_loader):
#             target = target.to(device)
#             _input = _input.to(device)

#             # compute output
#             output = net(_input)
#             loss = criterion(output, target)

#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
#             losses.update(loss.item(), _input.size(0))
#             top1.update(acc1[0].item(), _input.size(0))
#             top5.update(acc5[0].item(), _input.size(0))

#             t.set_postfix({
#                 'Loss': losses.avg,
#                 'Top1': top1.avg,
#                 'Top5': top5.avg
#             })
#             t.update(1)

# print('Loss:', losses.avg, '\t Top1:', top1.avg, '\t Top5:', top5.avg)

print("Proxyless-NAS")
if args.to_csv:
    try:
        df = pd.read_csv(args.to_csv)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Dataset', 'epsilon', 'alpha', 'iterations', 'Network', 'Clean Accuracy', 'FGSM', 'R-FGSM', 'PGD']) #'StepLL', 


if args.attacks == 'all' or args.attacks == 'clean':
    acc = test(net, testloader)

if args.attacks == 'all' or args.attacks == 'fgsm':
    fgsm = fgsm_attack_cifar(net, testloader, args.eps)

if args.attacks == 'all' or args.attacks == 'rfgsm':
    rfgsm = rfgsm_attack_cifar(net, testloader, args.eps, args.alpha, args.iters)

if args.attacks == 'all' or args.attacks == 'pgd':
    pgd = pgd_attack_cifar(net, testloader, args.eps, args.alpha, args.iters)

if args.attacks == 'all' or args.attacks == 'autoa':
    autoa = auto_attack_cifar(net, testloader)

# if args.attacks == 'all' or args.attacks == 'stepll':
#   stepll = stepll_attack_cifar(net, testloader, args.eps, args.alpha, args.iters)

dataset = 'cifar100' if args.cifar100 else 'cifar10'
if args.to_csv:
    vals = pd.Series({'Dataset': dataset, 'epsilon': args.eps, 'alpha':args.alpha, 'iterations':args.iters, 'Network': 'proxyless-nas', 'Clean Accuracy': acc, 'FGSM': fgsm, 'R-FGSM': rfgsm, 'PGD': pgd, 'AutoAttack': autoa})

    df = df.append(vals, ignore_index=True)
    df.to_csv(args.to_csv, index=False)
