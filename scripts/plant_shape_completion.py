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
from ssc_lmap.branch_completion import BranchCompletion
from ssc_lmap.grasp_planner import GraspPlanner
from data.arm_configs import home_angles, qlimits, home_pose


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset_dir', type=str, default='out_data/plant_assets')
    parser.add_argument('--obj_name', type=str, default='test_obj_1714573815')
    parser.add_argument('--trans_params_fn', type=str, default='data/tripod_finetune_params1.json')
    parser.add_argument('--close_branch_distance', type=float, default=0.05)
    parser.add_argument('--add_real_ur5', action='store_true')
    parser.add_argument('--ur5_ip', type=str, default="192.168.0.20")
    parser.add_argument('--sample_action_mode', type=str, default='3D-6directions')
    parser.add_argument('--pregrasp_dis', type=float, default=0.05)
    parser.add_argument('--out_data_dir', type=str, default='out_data')
    parser.add_argument('--exp_id', type=int, default=int(time.time()))
    parser.add_argument('--collision_threshold', type=float, default=10)
    parser.add_argument('--collision_distance', type=float, default=0.005)
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
    parser.add_argument('--num_cvx_samples', type=int, default=10)
    parser.add_argument('--object_boundary_threshold', type=float, default=0.0125)
    parser.add_argument('--move2branch_theta_limit', type=float, default=135)
    parser.add_argument('--graph_voxel_size', type=float, default=0.003)
    parser.add_argument('--deform_optimizer', type=str, default='adam')
    parser.add_argument('--deform_iters', type=int, default=20)
    parser.add_argument('--deform_lr', type=float, default=0.01)
    parser.add_argument('--energy_converge_threshold', type=float, default=0.01)
    parser.add_argument('--energy_converge_window', type=int, default=10)
    parser.add_argument('--num_box_pts', type=int, default=25)
    parser.add_argument('--nn_radius', type=float, default=0.020)
    parser.add_argument('--move_min_dist', type=float, default=0.01)
    parser.add_argument('--move_max_dist', type=float, default=0.05)
    parser.add_argument('--move_steps', type=int, default=3)
    parser.add_argument('--max_visible_pts_rate', type=float, default=0.95)
    args = parser.parse_args()


    # Load camera extrinsic parameters
    camera_params = json.load(open(args.trans_params_fn))
    cam2rob_trans = np.array(camera_params['cam2rob'])
    cam_center = cam2rob_trans[:3, 3]

    obj_data_dir = Path(f'{args.asset_dir}/{args.obj_name}')
    branch_pcd = o3d.io.read_point_cloud(str(obj_data_dir / f'{CLASSES[0]}_pcd.ply'))
    leaf_pcd = o3d.io.read_point_cloud(str(obj_data_dir / f'{CLASSES[1]}_pcd.ply'))
    fruit_pcd = o3d.io.read_point_cloud(str(obj_data_dir / f'{CLASSES[2]}_pcd.ply'))
    # o3d.visualization.draw_geometries_with_editing([leaf_pcd])

    # Get the largest component of the leaf point cloud using dbscan clustering
    leaf_labels = np.array(leaf_pcd.cluster_dbscan(eps=args.nn_radius, min_points=30, print_progress=False))
    nonneg_leaf_labels = leaf_labels[leaf_labels >= 0]
    max_num_label = np.argmax(np.bincount(nonneg_leaf_labels))
    leaf_pcd = leaf_pcd.select_by_index(np.where(leaf_labels == max_num_label)[0])
    o3d.visualization.draw_geometries([leaf_pcd])

    # Get the largest component of the fruit point cloud using dbscan clustering
    fruit_labels = np.array(fruit_pcd.cluster_dbscan(eps=args.nn_radius, min_points=30, print_progress=False))
    nonneg_fruit_labels = fruit_labels[fruit_labels >= 0]
    max_num_label = np.argmax(np.bincount(nonneg_fruit_labels))
    fruit_pcd = fruit_pcd.select_by_index(np.where(fruit_labels == max_num_label)[0])
    o3d.visualization.draw_geometries([fruit_pcd])

    raw_scene_pcd = o3d.io.read_point_cloud(str(obj_data_dir / f'crop_space_merge_pcd.ply'))

    mask_array = np.load(obj_data_dir / 'mask.npy')
    
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
    o3d.io.write_triangle_mesh(str(obj_data_dir / 'completed_branch_mesh.ply'), completed_branch_mesh)
    o3d.io.write_point_cloud(str(obj_data_dir / 'completed_branch_pcd.ply'), completed_branch_pcd)

    simplified_completed_branch_mesh = completed_branch_mesh.simplify_quadric_decimation(300)
    o3d.visualization.draw_geometries([completed_branch_pcd, branch_pcd, leaf_pcd, fruit_pcd, completed_branch_mesh])
    # o3d.visualization.draw_geometries([completed_branch_pcd, branch_pcd, leaf_pcd, fruit_pcd, simplified_completed_branch_mesh])
    
    """Compute free space points using octomap"""
    octomap_wrapper = OctomapWrapper()
    free_space_pts = octomap_wrapper.compute_free_space_points(raw_scene_pcd, fruit_pts, cam_center,
                                                               args.voxel_size, args.fruit_max_size,
                                                               args.free_space_bbx_scale, args.free_space_layer)
    free_space_pcd = o3d.geometry.PointCloud()
    free_space_pcd.points = o3d.utility.Vector3dVector(free_space_pts)
    
    o3d.visualization.draw_geometries([free_space_pcd, fruit_pcd, branch_pcd, leaf_pcd], window_name='Free Space PCD')
    o3d.io.write_point_cloud(str(obj_data_dir / 'free_space_pcd.ply'), free_space_pcd)
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
    o3d.io.write_triangle_mesh(str(obj_data_dir / 'fit_fruit_mesh.ply'), fit_fruit_mesh)
    """End of scene-consistent shape completion of fruit"""

    # save the shape completion results
    with open(obj_data_dir / 'fit_params.json', 'w') as f:
        fit_params = {
            'translation': out_dict['translation'].tolist(),
            'rotation': out_dict['rotation'].tolist(),
            'scale': out_dict['scale'].tolist()   
        }
        json.dump(fit_params, f, indent=2)
    torch.save(out_dict['latent'], f'{obj_data_dir}/fit_latent.pth')
    