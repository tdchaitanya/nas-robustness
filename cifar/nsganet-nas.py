import _init_paths
import os
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from nsga_net.misc import nsga_utils
from nsga_net.models import macro_genotypes
from nsga_net.models.macro_models import EvoNetwork
import nsga_net.models.micro_genotypes as genotypes
from nsga_net.models.micro_models import PyramidNetworkCIFAR as PyrmNASNet
import pandas as pd
from utils.net_utils import test
from utils.attacks_cifar import apgd_attack_cifar

parser = argparse.ArgumentParser(description='Adversarial attacks on CIFAR-10 (DARTS)')
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

parser.add_argument('-c', '--checkpoint_dir', default='../nsga_net/pretrained/', type=str,
                    help='path to pre-trained model weights')
parser.add_argument('--attacks', default='all', type=str,
                    help='specify which attack to run, default: all')
                    
args = parser.parse_args()

if args.gpu is not None:
    print(f'Using GPU: {args.gpu}')
    torch.cuda.set_device(args.gpu)

genotype = eval("genotypes.%s" % 'NSGANet')

if args.cifar100:
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
  ])

  testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
  path = os.path.join(args.checkpoint_dir, 'cifar100.pt')

  net = PyrmNASNet(36, num_classes=100, layers=20,
                         auxiliary=True, genotype=genotype,
                         increment=6, SE=True)
else:
  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
  path = os.path.join(args.checkpoint_dir, 'cifar10.pt')
  net = PyrmNASNet(26, num_classes=10, layers=20,
                         auxiliary=True, genotype=genotype,
                         increment=4, SE=True)

if args.gpu is not None:
  net = net.to(args.gpu)
  cudnn.benchmark=True
  net.droprate = 0.0

nsga_utils.load(net, path)

print("NSGA-Net")
apgd = apgd_attack_cifar(net, testloader, args.eps, args.iters, 10)
print("Average acc of AutoPGD-10 on NSGANet-NAS: {}".format(apgd))   

# if args.to_csv:
#     try:
#         df = pd.read_csv(args.to_csv)
#     except FileNotFoundError:
#         df = pd.DataFrame(columns=['Dataset', 'epsilon', 'alpha', 'iterations', 'Network', 'Clean Accuracy', 'FGSM', 'R-FGSM', 'StepLL', 'PGD'])

# if args.attacks == 'all' or args.attacks == 'clean':
#     acc = test(net, testloader)

# if args.attacks == 'all' or args.attacks == 'fgsm':
#     fgsm = fgsm_attack_cifar(net, testloader, args.eps)

# if args.attacks == 'all' or args.attacks == 'rfgsm':
#     rfgsm = rfgsm_attack_cifar(net, testloader, args.eps, args.alpha, args.iters)

# if args.attacks == 'all' or args.attacks == 'pgd':
#     pgd = pgd_attack_cifar(net, testloader, args.eps, args.alpha, args.iters)

# if args.attacks == 'all' or args.attacks == 'autoa':
#     autoa = auto_attack_cifar(net, testloader)



# dataset = 'cifar100' if args.cifar100 else 'cifar10'
# vals = pd.Series({'Dataset': dataset, 'epsilon': args.eps, 'alpha':args.alpha, 'iterations':args.iters, 'Network': 'nsganet', 'Clean Accuracy': acc, 'FGSM': fgsm, 'R-FGSM': rfgsm, 'PGD': pgd, 'AutoAttack': autoa})

# df = df.append(vals, ignore_index=True)
# df.to_csv(args.to_csv, index=False)
