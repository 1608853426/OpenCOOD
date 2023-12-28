import argparse
import os
import time
from tqdm import tqdm
import argparse
import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt



def get_data_loader():
    opt = get_where2comm_opt()
    hypes = yaml_utils.load_yaml(None, opt)
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    return data_loader, opencood_dataset

def get_where2comm_opt():
    opt = argparse.Namespace()
    opt.model_dir = '/root/project/OpenCOOD/opencood/logs/point_pillar_where2comm_v2xset/'
    """ opt['fusion_method'] = 'intermediate'
    opt['show_vis'] = False
    opt['show_sequence'] = False
    opt['save_vis'] = False
    opt['save_npy'] = False
    opt['global_sort_detections'] = False """
    opt.fusion_method = 'intermediate'
    opt.show_vis = False
    opt.show_sequence = False
    opt.save_vis = False
    opt.save_npy = False
    opt.global_sort_detections = False
    return opt
    
class Where2commData():
    def __init__(self):
        self.data_loader, self.opencood_dataset = get_data_loader()
        self.opt = get_where2comm_opt()
        self.hypes = yaml_utils.load_yaml(None, self.opt)
        
    
    def get_idx_data(self, idx):
        # 获取data_loader中的第idx个数据
        data = None
        for i, batch_data in enumerate(self.data_loader):
            if i == idx:
                data = batch_data
                break
        return data
    
    def get_opt(self):
        return self.opt
    
    def get_hypes(self):
        return self.hypes