# Download the dataste and unzip it in data directory of the root folder
# !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip -qq tiny-imagenet-200.zip  -qq && rm tiny-imagenet-200.zip

import pandas as pd
import os
import shutil
import glob
import copy

categories = os.listdir('../data/tiny-imagenet-200/train/')
assert len(categories) == 200
for each in categories:
    os.mkdir(f'../data/tiny-imagenet-200/val/{each}')

df = pd.read_csv('t../data/iny-imagenet-200/val/val_annotations.txt', delimiter='\t', header=None)

label_to_cat = dict(zip(df[0], df[1]))

for each in glob.glob('../data/tiny-imagenet-200/val/images/*.JPEG'):
    src = copy.copy(each)
    fl_name = each.split('/')[-1]
    dest = each.replace('images', label_to_cat[fl_name])
    shutil.move(src, dest)
    
#!rm -rf tiny-imagenet-200/val/images/ tiny-imagenet-200/val/val_annotations.txt tiny-imagenet-200/test/