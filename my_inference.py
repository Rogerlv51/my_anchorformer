import numpy as np
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import parser
import time
from utils.logger import *
from utils.config import *
from datasets.io import IO
import open3d as o3d
from datasets.data_transforms import Compose
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

def my_normalize(pc):
    min_val = np.min(pc, axis=0)
    max_val = np.max(pc, axis=0)
    range_val = max_val - min_val
    point_cloud_normalized = pc - min_val
    point_cloud_normalized /= range_val
    point_cloud_normalized -= 0.5
    return point_cloud_normalized, min_val, max_val

def my_denormalize(point_cloud_normalized, min_val, max_val):
    range_val = max_val - min_val
    point_cloud_denormalized = point_cloud_normalized + 0.5
    point_cloud_denormalized *= range_val
    point_cloud_denormalized += min_val
    return point_cloud_denormalized

def test_net(args, config, file_path):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
 
    base_model = builder.model_builder(config.model)
    print_log(base_model, logger = logger)
    
    # load checkpoints
    
    builder.load_model(base_model, args.ckpts)
    
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()
    data = IO.get(file_path).astype(np.float32)
    data, min_val, max_val = my_normalize(data)
    
    transform = Compose([{
        'callback': 'RandomSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': data})
    test(base_model, pc_ndarray_normalized['input'].unsqueeze(0), min_val, max_val)

def test(base_model, data, min_val, max_val):

    base_model.eval()  # set model to eval mode
    
    with torch.no_grad():
        partial = data.cuda()
        target_path = "."            
        ret = base_model(partial)
        coarse_points = ret[0]
        dense_points = ret[1].squeeze(0).detach().cpu().numpy()
        dense_points = my_denormalize(dense_points, min_val, max_val)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_points)
        o3d.visualization.draw_geometries([pcd], height=680, width=1080, window_name="Pred")
        o3d.io.write_point_cloud(os.path.join(target_path, "point_cloud.ply"), pcd)



if __name__ == '__main__':
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    if args.launcher == 'none':
        args.distributed = False

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')    
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    config = get_config(args, logger = logger)

    path = '2023-08-04_16_41_40-Teeth-31.ply'
    test_net(args, config, path)