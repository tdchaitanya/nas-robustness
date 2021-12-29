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
from pcdarts import genotypes
from pcdarts.model import NetworkImageNet as Network
import pandas as pd
from utils.net_utils import test_imnet
from utils.attacks_imnet import fgsm_attack_imnet, rfgsm_attack_imnet, stepll_attack_imnet, pgd_attack_imnet

parser = argparse.ArgumentParser(description='Adversarial attacks on ImageNet')
parser.add_argument('--data', default ='../data/Imagenet',
                    help='path to dataset')
parser.add_argument('--to-csv', default=None,
                    help='path to dataset')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='batch size to use (on test set)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--eps', default=0.007, type=float,
                    help='Epsilon value for attack, strenght of perturbation')
parser.add_argument('--iters', default=3, type=int,
                    help='number of attack iterations')
parser.add_argument('--alpha', default=0.03, type=float,
                    help='number of attack iterations')
parser.add_argument('-c', '--checkpoint_dir', default='../pcdarts/pretrained/', type=str,
                    help='path to pre-trained model weights')

args = parser.parse_args()


if args.gpu is not None:
    print(f'Using GPU: {args.gpu}')
    torch.cuda.set_device(args.gpu)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

testset = torchvision.datasets.ImageFolder(root=f'{args.data}/valid/', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

genotype = eval("genotypes.%s" % 'PC_DARTS_image')
net = Network(48, 1000, 14, True, genotype)
if args.gpu is not None:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
path = os.path.join(args.checkpoint_dir, 'imagenet.pt')
checkpoint = torch.load(path, map_location=device)['state_dict']
net.load_state_dict(checkpoint)
net.drop_path_prob = 0.2

if args.gpu is not None:
  net = net.to(args.gpu)
  cudnn.benchmark=True

print("PC-DARTS")
if args.to_csv:
    try:
        df = pd.read_csv(args.to_csv)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Dataset', 'epsilon', 'alpha', 'iterations', 'Network',
                                   'Clean Accuracy-Top1', 'Clean Accuracy-Top5', 'FGSM-Top1', 'FGSM-Top5',
                                   'R-FGSM-Top1', 'R-FGSM-Top5', 'StepLL-Top1', 'StepLL-Top5', 'PGD-Top1', 'PGD-Top5'])
clean_top1, clean_top5 = test_imnet(net, testloader)

fgsm_top1, fgsm_top5 = fgsm_attack_imnet(net, testloader,  args.eps)

rfgsm_top1, rfgsm_top5 = rfgsm_attack_imnet(net, testloader,  args.eps, args.alpha, args.iters)

stepll_top1, stepll_top5 = stepll_attack_imnet(net, testloader,  args.eps, args.alpha, args.iters)

pgd_top1, pgd_top5 = pgd_attack_imnet(net, testloader,  args.eps, args.alpha, args.iters)


vals = pd.Series({'Dataset': 'imagenet',
                  'epsilon': args.eps,
                  'alpha':args.alpha,
                  'iterations':args.iters,
                  'Network': 'pc-darts',
                  'Clean Accuracy-Top1': clean_top1,
                  'Clean Accuracy-Top5': clean_top5,
                  'FGSM-Top1': fgsm_top1,
                  'FGSM-Top5': fgsm_top5,
                  'R-FGSM-Top1': rfgsm_top1,
                  'R-FGSM-Top5': rfgsm_top5,
                  'StepLL-Top1': stepll_top1,
                  'StepLL-Top5': stepll_top5,
                  'PGD-Top1': pgd_top1,
                  'PGD-Top5': pgd_top5})

df = df.append(vals, ignore_index=True)
df.to_csv(args.to_csv, index=False)

