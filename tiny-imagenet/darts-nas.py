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
from darts import genotypes
from darts.model import NetworkImageNet as Network

from utils.net_utils import test_imnet
from utils.attacks_imnet import fgsm_attack_imnet, rfgsm_attack_imnet, stepll_attack_imnet, pgd_attack_imnet

parser = argparse.ArgumentParser(description='Adversarial attacks on Tiny Imagenet')
parser.add_argument('--data', default ='../data/tiny-imagenet',
                    help='path to dataset')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='batch size to use (on test set)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')

parser.add_argument('-c', '--checkpoint_dir', default='../darts/pretrained/', type=str,
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

genotype = eval("genotypes.%s" % 'DARTS')
net = Network(48, 1000, 14, True, genotype)
net.drop_path_prob = 0.0
path = os.path.join(args.checkpoint_dir, 'imagenet.pt')
checkpoint = torch.load(path, map_location=device)['state_dict']
net.load_state_dict(checkpoint)

if args.gpu is not None:
  net = net.to(args.gpu)
  cudnn.benchmark=True


test_imnet(net, testloader)

fgsm_attack_imnet(net, testloader)

rfgsm_attack_imnet(net, testloader)

stepll_attack_imnet(net, testloader)

pgd_attack_imnet(net, testloader)
