import os.path as osp
import sys

def add_path(path):
  if path not in sys.path:
    sys.path.append(path)

this_dir = osp.dirname(__file__)

root_path = osp.abspath(osp.join(this_dir,".."))
add_path(root_path)

darts_path = osp.join(root_path, "darts")
add_path(darts_path)

pdarts_path = osp.join(root_path, "pdarts")
add_path(pdarts_path)

nsga_path = osp.join(root_path, "nsga_net")
add_path(nsga_path)

pcdarts_path = osp.join(root_path, "pcdarts")
add_path(pcdarts_path)

proxyless_path = osp.join(root_path, "proxylessnas")
add_path(proxyless_path)

utils_path = osp.join(root_path, "utils")
add_path(utils_path)


