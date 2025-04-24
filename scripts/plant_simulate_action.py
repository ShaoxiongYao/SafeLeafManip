import time
import numpy as np
import open3d as o3d
from pathlib import Path
import torch
import json

try:
    import octomap
except:
    print('INFO: octomap-python not installed')
    pass
import copy

import context
from ssc_lmap.segment_plant import CLASSES
from ssc_lmap.grasp_planner import GraspPlanner, GraspPlannerConfig, PullActionConfig
from ssc_lmap.octomap_wrapper import OctomapWrapper
from ssc_lmap.vis_utils import vis_grasp_frames

from ssc_lmap.pts_utils import trans_matrix2pose, get_largest_dbscan_component, get_discrete_move_directions
from ssc_lmap.embed_deform_graph import NodeGraph, make_embed_deform_graph, PlantSimulatorConfig
from yaml import safe_load
import dacite


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_dir', type=str, help='plant segmentation directory, should contains fruit, leaf, branch pcd')
    parser.add_argument('--shape_complete_dir', type=str, help='plant shape completion output directory')
    parser.add_argument('--trans_params_fn', type=str, default='data/tripod_finetune_params1.json')
    parser.add_argument('--simulate_action_config', type=str, default='configs/simulate_action.yaml',
                        help='plant action simulation config file describing the parameters for simulation')
    parser.add_argument('--sim_out_dir', type=str, default='data/sim_out', help='output directory for simulation')
    args = parser.parse_args()
    
    time_stats_dict = {}
    
    # Load camera extrinsic parameters
    camera_params = json.load(open(args.trans_params_fn))
    cam2rob_trans = np.array(camera_params['cam2rob'])
    cam_center = cam2rob_trans[:3, 3]
    
    # Prepare configurations for shape completion
    simulate_action_config = safe_load(open(args.simulate_action_config, 'r'))
    preprocess_config = simulate_action_config['preprocess']
    grasp_planner_config = dacite.from_dict(GraspPlannerConfig, simulate_action_config['grasp_planner'])
    simulator_config = dacite.from_dict(PlantSimulatorConfig, simulate_action_config['plant_simulator'])
    octomap_config = simulate_action_config['octomap']
    action_config = dacite.from_dict(PullActionConfig, simulate_action_config['action'])
    view_params = simulate_action_config['view_params']

    segment_dir = Path(args.segment_dir)
    shape_complete_dir = Path(args.shape_complete_dir)
    # Load segmented point clouds
    branch_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[0]}_pcd.ply'))
    leaf_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[1]}_pcd.ply'))
    fruit_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[2]}_pcd.ply'))
    cropped_scene_pcd = o3d.io.read_point_cloud(str(segment_dir / f'cropped_scene_pcd.ply'))
    
    """Preprocess leaf and fruit point clouds"""
    leaf_pcd = get_largest_dbscan_component(leaf_pcd, nn_radius=preprocess_config['dbscan_eps'], 
                                            min_points=preprocess_config['dbscan_min_points'], vis_pcd=False)
    fruit_pcd = get_largest_dbscan_component(fruit_pcd, nn_radius=preprocess_config['dbscan_eps'], 
                                             min_points=preprocess_config['dbscan_min_points'], vis_pcd=False)
    """End of preprocessing leaf and fruit point clouds"""
    
    # Load completed triangle meshes
    fit_fruit_mesh = o3d.io.read_triangle_mesh(str(shape_complete_dir / 'completed_fruit_mesh.ply'))
    fit_fruit_pcd = fit_fruit_mesh.sample_points_poisson_disk(number_of_points=10000)

    """Plan grasps on segmented plant leaf"""
    grasp_sample_start_time = time.time()
    grasp_planner = GraspPlanner(branch_pcd, config=grasp_planner_config)
    grasp_frame_lst = grasp_planner.plan_grasp_leaf(leaf_pcd, vis_cvx_hull=False, vis_approach_pts=False, 
                                                    vis_grasp_frame=False, vis_remove_outliers=False)
    print('number of grasp frames:', len(grasp_frame_lst))

    # Visualize grasp frames
    grasp_geom_list = vis_grasp_frames(grasp_frame_lst)
    o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, fruit_pcd] + grasp_geom_list, **view_params)
    """End of planning grasps"""

    sim_out_dir = Path(args.sim_out_dir)
    sim_out_dir.mkdir(parents=True, exist_ok=True)
    
    all_move_dir_ary = get_discrete_move_directions(action_config.move_type)
    all_move_dir_tsr = torch.from_numpy(all_move_dir_ary).double().to('cuda')
    
    octomap_wrapper = OctomapWrapper()

    final_energy_dict = {}
    final_number_visible_pts_dict = {}
    for grasp_id, grasp_frame in enumerate(grasp_frame_lst):
        
        # Prepare embedded deformation graph simulation
        node_graph = make_embed_deform_graph(grasp_frame, branch_pcd, leaf_pcd, 
                                             simulator_config.num_box_pts, simulator_config.close_branch_distance,
                                             simulator_config.graph_voxel_size, simulator_config.nn_radius)

        rest_pcd = node_graph.get_pcd()
        rest_pcd.paint_uniform_color([0.7, 0.0, 0.7])        
        line_set = node_graph.get_line_set()
        o3d.visualization.draw_geometries([line_set])

        for move_id, move_dir_tsr in enumerate(all_move_dir_tsr):

            # reset the state of the graph to the initial state
            node_graph.reset_state()

            for move_step_length in np.linspace(action_config.move_min_dist, 
                                                action_config.move_max_dist, 
                                                num=action_config.move_steps):

                np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_move_vector.npy', 
                        (move_step_length*move_dir_tsr).cpu().numpy())

                """Run embedded deformation graph simulation"""
                # Prepare moved handle points for simulation
                # TODO: find a better way to indicate movable handle points
                handle_pts_tsr = node_graph.rest_pts_tsr[node_graph.handle_idx].clone()
                handle_pts_tsr[:simulator_config.num_box_pts] += move_step_length * move_dir_tsr

                with torch.no_grad():
                    energy_list = node_graph.solve_global_local(node_graph.handle_idx, handle_pts_tsr, simulator_config.deform_iters,
                                                                simulator_config.energy_converge_threshold, verbose=True)
                energy = node_graph.energy(node_graph.handle_idx, handle_pts_tsr)
                final_energy_dict[f'{grasp_id:02d}_{move_id:02d}'] = energy.item()
                """End of run embedded deformation graph simulation"""
                
                # Visualize deformed graph
                curr_sim_pcd = node_graph.get_pcd(node_graph.handle_idx, handle_pts_tsr, [0.0, 1.0, 0.0])
                arrow_lst = node_graph.get_deform_arrow_lst(node_graph.handle_idx, handle_pts_tsr)
                o3d.visualization.draw_geometries([rest_pcd, curr_sim_pcd] + arrow_lst)

                curr_vis_pcd = node_graph.get_vis_pcd(fix_pts_idx=np.arange(len(branch_pcd.points)))
                o3d.visualization.draw_geometries([rest_pcd, curr_sim_pcd, curr_vis_pcd] + arrow_lst)
                merged_pcd = curr_vis_pcd + fit_fruit_pcd #+ arm_pcd
                # o3d.visualization.draw_geometries([merged_pcd, fruit_pcd, leaf_pcd, branch_pcd, arm_pcd])
                
                """Ray casting to compute visible points"""
                vis_points, octo_points = octomap_wrapper.compute_visible_points(merged_pcd, fit_fruit_pcd, cam_center,
                                                                                 octomap_config['voxel_size'], verbose=True)
                print('ray casting number visible points:', len(vis_points))
                final_number_visible_pts_dict[f'{grasp_id:02d}_{move_id:02d}'] = len(vis_points)
                """End of ray casting to compute visible points"""

                # if len(vis_points) >= all_vis_fit_fruit_pts_num * args.max_visible_pts_rate:
                #     print('INFO: reach maximum visible points rate:', len(vis_points) / all_vis_fit_fruit_pts_num)
                #     # break
    
    with open(sim_out_dir / 'final_energy.json', 'w') as f:
        json.dump(final_energy_dict, f, indent=2)
    
    with open(sim_out_dir / 'final_number_visible_pts.json', 'w') as f:
        json.dump(final_number_visible_pts_dict, f, indent=2)
