import torch, pypose as pp
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import time
from pathlib import Path
from yaml import safe_load
import json

import dacite
import context
from ssc_lmap.segment_plant import CLASSES
from ssc_lmap.scene_consistent_deepsdf import SceneConsistentDeepSDF, FruitCompletionConfig
from ssc_lmap.octomap_wrapper import OctomapWrapper, FreeSpaceConfig
from ssc_lmap.pts_utils import get_largest_dbscan_component
from ssc_lmap.branch_completion import BranchCompletion, BranchCompletionConfig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_dir', type=str, help='plant segmentation directory, should contains fruit, leaf, branch pcd')
    parser.add_argument('--shape_complete_dir', type=str, help='plant shape completion output directory')
    parser.add_argument('--shape_complete_config', type=str, default='configs/plant_shape_completion.yaml', 
                        help='plant shape completion config file describing the parameters for the completion')
    parser.add_argument('--trans_params_fn', type=str, default='data/tripod_finetune_params1.json')
    args = parser.parse_args()


    # Load camera extrinsic parameters
    camera_params = json.load(open(args.trans_params_fn))
    cam2rob_trans = np.array(camera_params['cam2rob'])
    cam_center = cam2rob_trans[:3, 3]

    # Prepare input and output directories
    segment_dir = Path(args.segment_dir)
    shape_complete_dir = Path(args.shape_complete_dir)
    shape_complete_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare configurations for shape completion
    shape_complete_config = safe_load(open(args.shape_complete_config, 'r'))
    preprocess_config = shape_complete_config['preprocess']
    free_space_config = dacite.from_dict(FreeSpaceConfig, shape_complete_config['free_space'])
    branch_completion_config = dacite.from_dict(BranchCompletionConfig, shape_complete_config['branch_completion'])
    fruit_completion_config = dacite.from_dict(FruitCompletionConfig, shape_complete_config['fruit_completion'])

    # Load segmented point clouds
    branch_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[0]}_pcd.ply'))
    leaf_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[1]}_pcd.ply'))
    fruit_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[2]}_pcd.ply'))
    cropped_scene_pcd = o3d.io.read_point_cloud(str(segment_dir / f'cropped_scene_pcd.ply'))

    # Preprocess leaf and fruit point clouds
    leaf_pcd = get_largest_dbscan_component(leaf_pcd, nn_radius=preprocess_config['dbscan_eps'], 
                                            min_points=preprocess_config['dbscan_min_points'], vis_pcd=False)
    fruit_pcd = get_largest_dbscan_component(fruit_pcd, nn_radius=preprocess_config['dbscan_eps'], 
                                            min_points=preprocess_config['dbscan_min_points'], vis_pcd=False)

    fruit_pts = np.array(fruit_pcd.points)
    leaf_pts = np.array(leaf_pcd.points)
    branch_pts = np.array(branch_pcd.points)

    time_stats_dict = {}
    
    # Fit branch mesh using deformation cylinder
    branch_completor = BranchCompletion(cylinder_resolution = branch_completion_config.cylinder_resolution)
    start_time = time.time()    
    completed_branch_mesh = branch_completor.complete(branch_pcd, cam_center)
    completed_branch_pcd = completed_branch_mesh.sample_points_poisson_disk(number_of_points=5000)
    end_time = time.time()
    time_stats_dict['branch_completion_time'] = end_time - start_time
    o3d.io.write_triangle_mesh(str(shape_complete_dir / 'completed_branch_mesh.ply'), completed_branch_mesh)

    o3d.visualization.draw_geometries([completed_branch_pcd, branch_pcd, leaf_pcd, fruit_pcd, completed_branch_mesh])
    
    """Compute free space points using octomap"""
    octomap_wrapper = OctomapWrapper()
    free_space_pts = octomap_wrapper.compute_free_space_points(cropped_scene_pcd, fruit_pts, cam_center,
                                                               free_space_config.voxel_size, 
                                                               free_space_config.fruit_max_size,
                                                               free_space_config.free_space_bbx_scale, 
                                                               free_space_config.free_space_layer)
    free_space_pcd = o3d.geometry.PointCloud()
    free_space_pcd.points = o3d.utility.Vector3dVector(free_space_pts)
    
    o3d.visualization.draw_geometries([free_space_pcd, fruit_pcd, branch_pcd, leaf_pcd], window_name='Free Space PCD')
    o3d.io.write_point_cloud(str(shape_complete_dir / 'free_space_pcd.ply'), free_space_pcd)
    """End of compute free space points using octomap"""

    
    """Scene-consistent shape completion of fruit"""
    # load deep sdf decoder and init latent code
    semantic_deepsdf = SceneConsistentDeepSDF(fruit_completion_config, checkpoint='latest')
    out_dict = semantic_deepsdf.fit_plant_pcd(fruit_pcd, completed_branch_pcd, branch_pcd, 
                                              leaf_pcd, cam_center, free_space_pcd, verbose=True, vis=True)
    time_stats_dict['fit_time'] = out_dict['time']
    fit_fruit_mesh = out_dict['final_shape_mesh']
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
    