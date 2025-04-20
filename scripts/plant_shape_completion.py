import torch, pypose as pp
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import time
from pathlib import Path
import pyrealsense2 as rs
import sklearn.neighbors as skn
from pypose.optim.strategy import Constant

try:
    import octomap
except:
    print('INFO: octomap-python not installed')
    pass
import copy

import context
from ssc_lmap.segment_plant import CLASSES
from ssc_lmap.embed_deform_graph import NodeGraph
from ssc_lmap.robot_control.ur5_sim import UR5Sim
from ssc_lmap.robot_control.ur5_real import UR5Real
from ssc_lmap.semantic_deepsdf_completion import SemanticDeepSDFCompletion
from ssc_lmap.octomap_wrapper import OctomapWrapper

import tqdm
import json
from copy import deepcopy


import context
from ssc_lmap.vis_utils import create_arrow_lst, gen_trans_box, bool2color, image_plane2pcd, create_ball
from ssc_lmap.pts_utils import get_largest_dbscan_component
from ssc_lmap.branch_completion import BranchCompletion
from ssc_lmap.grasp_planner import GraspPlanner
from data.arm_configs import home_angles, qlimits, home_pose


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_dir', type=str, help='plant segmentation directory, should contains fruit, leaf, branch pcd')
    parser.add_argument('--shape_complete_dir', type=str, help='plant shape completion output directory')
    parser.add_argument('--trans_params_fn', type=str, default='data/tripod_finetune_params1.json')
    parser.add_argument('--voxel_size', type=float, default=0.003)
    parser.add_argument('--free_space_layer', type=int, default=1)
    parser.add_argument('--free_space_bbx_scale', type=float, default=1.5)
    parser.add_argument('--fruit_min_size', type=float, default=0.01)
    parser.add_argument('--fruit_max_size', type=float, default=0.05)
    parser.add_argument('--NUM_ITERS', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_correspondence', type=int, default=10)
    parser.add_argument('--weight_surface', type=float, default=1.0)
    parser.add_argument('--weight_negative', type=float, default=2.0)
    parser.add_argument('--weight_peduncle', type=float, default=0.1)
    parser.add_argument('--weight_regularization', type=float, default=0.001)
    parser.add_argument('--peduncle_tolerance', type=float, default=0.0001)
    parser.add_argument('--weight_neg_com_branch', type=float, default=0.1)
    parser.add_argument('--opt_mesh_res', type=int, default=16)
    parser.add_argument('--output_mesh_res', type=int, default=128)
    parser.add_argument('--nn_radius', type=float, default=0.020)
    args = parser.parse_args()


    # Load camera extrinsic parameters
    camera_params = json.load(open(args.trans_params_fn))
    cam2rob_trans = np.array(camera_params['cam2rob'])
    cam_center = cam2rob_trans[:3, 3]

    # Prepare input and output directories
    segment_dir = Path(args.segment_dir)
    shape_complete_dir = Path(args.shape_complete_dir)
    shape_complete_dir.mkdir(parents=True, exist_ok=True)

    branch_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[0]}_pcd.ply'))
    leaf_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[1]}_pcd.ply'))
    fruit_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[2]}_pcd.ply'))
    cropped_scene_pcd = o3d.io.read_point_cloud(str(segment_dir / f'cropped_scene_pcd.ply'))

    # Preprocess leaf and fruit point clouds
    leaf_pcd = get_largest_dbscan_component(leaf_pcd, nn_radius=args.nn_radius, min_points=30, vis_pcd=False)
    fruit_pcd = get_largest_dbscan_component(fruit_pcd, nn_radius=args.nn_radius, min_points=30, vis_pcd=False)

    
    fruit_pts = np.array(fruit_pcd.points)
    leaf_pts = np.array(leaf_pcd.points)
    branch_pts = np.array(branch_pcd.points)

    time_stats_dict = {}
    
    # Fit branch mesh using deformation cylinder
    branch_completor = BranchCompletion(cylinder_resolution = 0.003)
    start_time = time.time()    
    completed_branch_mesh = branch_completor.complete(branch_pcd, cam_center)
    completed_branch_pcd = completed_branch_mesh.sample_points_poisson_disk(number_of_points=5000)
    end_time = time.time()
    print('Time taken:', end_time - start_time)
    time_stats_dict['branch_completion_time'] = end_time - start_time
    o3d.io.write_triangle_mesh(str(shape_complete_dir / 'completed_branch_mesh.ply'), completed_branch_mesh)

    simplified_completed_branch_mesh = completed_branch_mesh.simplify_quadric_decimation(300)
    o3d.visualization.draw_geometries([completed_branch_pcd, branch_pcd, leaf_pcd, fruit_pcd, completed_branch_mesh])
    # o3d.visualization.draw_geometries([completed_branch_pcd, branch_pcd, leaf_pcd, fruit_pcd, simplified_completed_branch_mesh])
    
    """Compute free space points using octomap"""
    octomap_wrapper = OctomapWrapper()
    free_space_pts = octomap_wrapper.compute_free_space_points(cropped_scene_pcd, fruit_pts, cam_center,
                                                               args.voxel_size, args.fruit_max_size,
                                                               args.free_space_bbx_scale, args.free_space_layer)
    free_space_pcd = o3d.geometry.PointCloud()
    free_space_pcd.points = o3d.utility.Vector3dVector(free_space_pts)
    
    o3d.visualization.draw_geometries([free_space_pcd, fruit_pcd, branch_pcd, leaf_pcd], window_name='Free Space PCD')
    o3d.io.write_point_cloud(str(shape_complete_dir / 'free_space_pcd.ply'), free_space_pcd)
    """End of compute free space points using octomap"""

    
    """Scene-consistent shape completion of fruit"""
    # load deep sdf decoder and init latent code
    DeepSDF_DIR = 'ssc_lmap/HortiMapping/deepsdf/models/sweetpepper_32'
    semantic_deepsdf = SemanticDeepSDFCompletion(DeepSDF_DIR, checkpoint='latest')

    weight_dict = {
        'weight_surface': args.weight_surface,
        'weight_negative': args.weight_negative,
        'weight_peduncle': args.weight_peduncle,
        'weight_regularization': args.weight_regularization,
        'peduncle_tolerance' : args.peduncle_tolerance,
        'weight_neg_com_branch': args.weight_neg_com_branch
    }
    out_dict = semantic_deepsdf.fit_plant_pcd(fruit_pcd, completed_branch_pcd, branch_pcd, leaf_pcd, cam_center, free_space_pcd, 
                                              args.opt_mesh_res, fruit_min_size=args.fruit_min_size, 
                                              # args.output_mesh_res, fruit_min_size=args.fruit_min_size, 
                                              num_correspondence=args.num_correspondence, 
                                              weight_dict=weight_dict, NUM_ITERS=args.NUM_ITERS,
                                              lr=args.lr, output_mesh_res=args.output_mesh_res, verbose=True, vis=True)
    fit_fruit_mesh = out_dict['final_shape_mesh']
    time_stats_dict['fit_time'] = out_dict['time']
    o3d.io.write_triangle_mesh(str(shape_complete_dir / 'completed_fruit_mesh.ply'), fit_fruit_mesh)
    """End of scene-consistent shape completion of fruit"""

    # save the shape completion results
    with open(shape_complete_dir / 'completed_fruit_sim3_params.json', 'w') as f:
        sim3_params = {
            'translation': out_dict['translation'].tolist(),
            'rotation': out_dict['rotation'].tolist(),
            'scale': out_dict['scale'].tolist()   
        }
        json.dump(sim3_params, f, indent=2)
    torch.save(out_dict['latent'], shape_complete_dir / 'completed_fruit_latent.pth' )
    