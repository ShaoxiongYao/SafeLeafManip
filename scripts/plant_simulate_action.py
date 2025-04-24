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
from ssc_lmap.grasp_planner import GraspPlanner, GraspPlannerConfig
from ssc_lmap.octomap_wrapper import OctomapWrapper
from ssc_lmap.vis_utils import create_ball, create_arrow_lst, bool2color
from ssc_lmap.vis_utils import gen_trans_box, image_plane2pcd, create_ball, vis_grasp_frames

from ssc_lmap.pts_utils import trans_matrix2pose, get_largest_dbscan_component, get_discrete_move_directions
from ssc_lmap.embed_deform_graph import NodeGraph, make_embed_deform_graph, PlantSimulatorConfig
from yaml import safe_load
import dacite


view_params = {	
    "front" : [ -0.72719746380345296, -0.65442554290856003, 0.20714984293178301 ],
    "lookat" : [ -0.16539800516476486, 0.737540449204104, 0.41621635030538628 ],
    "up" : [ 0.10859730785732971, 0.18829519911886816, 0.97608992552680629 ],
    "zoom" : 0.21999999999999958
}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_dir', type=str, help='plant segmentation directory, should contains fruit, leaf, branch pcd')
    parser.add_argument('--shape_complete_dir', type=str, help='plant shape completion output directory')
    parser.add_argument('--simulate_action_config', type=str, default='configs/simulate_action.yaml',
                        help='plant action simulation config file describing the parameters for simulation')
    parser.add_argument('--sim_out_dir', type=str, default='data/sim_out', help='output directory for simulation')
    parser.add_argument('--move_min_dist', type=float, default=0.01)
    parser.add_argument('--move_max_dist', type=float, default=0.05)
    parser.add_argument('--move_steps', type=int, default=3)
    parser.add_argument('--voxel_size', type=float, default=0.003)
    
    parser.add_argument('--trans_params_fn', type=str, default='data/tripod_finetune_params1.json')
    
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
    # free_space_config = dacite.from_dict(FreeSpaceConfig, shape_complete_config['free_space'])
    # branch_completion_config = dacite.from_dict(BranchCompletionConfig, shape_complete_config['branch_completion'])
    # fruit_completion_config = dacite.from_dict(FruitCompletionConfig, shape_complete_config['fruit_completion'])
    

    segment_dir = Path(args.segment_dir)
    shape_complete_dir = Path(args.shape_complete_dir)
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
    
    # Load completed triangle meshes
    fit_fruit_mesh = o3d.io.read_triangle_mesh(str(shape_complete_dir / 'completed_fruit_mesh.ply'))
    fit_fruit_pcd = fit_fruit_mesh.sample_points_poisson_disk(number_of_points=10000)

    grasp_sample_start_time = time.time()
    grasp_planner = GraspPlanner(branch_pcd, config=grasp_planner_config)
    grasp_frame_lst = grasp_planner.plan_grasp_leaf(leaf_pcd, vis_cvx_hull=False, vis_approach_pts=False, 
                                                    vis_grasp_frame=False, vis_remove_outliers=False)
    print('number of grasp frames:', len(grasp_frame_lst))

    # Visualize grasp frames
    grasp_geom_list = vis_grasp_frames(grasp_frame_lst)
    o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, fruit_pcd] + grasp_geom_list, **view_params)

    sim_out_dir = Path(args.sim_out_dir)
    sim_out_dir.mkdir(parents=True, exist_ok=True)
    
    move_vec_ary = get_discrete_move_directions('3D-6directions')
    
    octomap_wrapper = OctomapWrapper()

    final_energy_lst = []
    final_number_visible_pts_lst = []
    for grasp_id, grasp_frame in enumerate(grasp_frame_lst):

        precompute_start_time = time.time()
        
        # Prepare embedded deformation graph simulation
        node_graph = make_embed_deform_graph(grasp_frame, branch_pcd, leaf_pcd, 
                                             simulator_config.num_box_pts, simulator_config.close_branch_distance,
                                             simulator_config.graph_voxel_size, simulator_config.nn_radius)

        rest_pcd = node_graph.get_pcd()
        rest_pcd.paint_uniform_color([0.7, 0.0, 0.7])

        print('number of nodes:', node_graph.num_pts)
        print('number of edges:', node_graph.num_edges)
        
        all_vis_pts = node_graph.vis_pts
        all_sim_pts = node_graph.rest_pts_tsr.cpu().numpy()
        handle_idx = node_graph.handle_idx
        
        vis_beta = node_graph.get_pts_beta(all_vis_pts, rbf_sig=0.3, dist_max=0.05)

        line_set = node_graph.get_line_set()
        o3d.visualization.draw_geometries([line_set])
        
        precompute_end_time = time.time()
        print('precompute time:', precompute_end_time - precompute_start_time)
        time_stats_dict[f'grasp_{grasp_id:02d}_precompute_time'] = precompute_end_time - precompute_start_time

        for move_id, local_move_vec in enumerate(move_vec_ary):

            world_move_vec = local_move_vec.copy()

            np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_local_move_vec.npy', local_move_vec)
            print('local move vec:', local_move_vec)

            # reset the state of the graph to the initial state
            node_graph.reset_state()

            for move_step_length in np.linspace(args.move_min_dist, args.move_max_dist, num=args.move_steps):

                move_start_time = time.time()

                np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_world_move_vec.npy', 
                        move_step_length*world_move_vec)

                rest_pts_tsr = torch.tensor(all_sim_pts[handle_idx, :], dtype=torch.double, device='cuda')
                energy = node_graph.energy(handle_idx, rest_pts_tsr)
                print('initial energy:', energy.item())
                
                current_points = all_sim_pts[handle_idx].copy()
                current_points[:simulator_config.num_box_pts] += move_step_length * world_move_vec
                handle_pts_tsr = torch.tensor(current_points, dtype=torch.double, device='cuda')

                arrow_lst = create_arrow_lst(all_sim_pts[handle_idx], handle_pts_tsr.detach().cpu().numpy())
                # o3d.visualization.draw_geometries([branch_pcd, completed_branch_mesh, leaf_pcd, box_start_pcd, grasp_coord_frame] + arrow_lst)

                geom = node_graph.get_pcd(handle_idx=handle_idx, handle_pts_tsr=handle_pts_tsr)
                # o3d.visualization.draw_geometries([rest_pcd, geom] + arrow_lst)

                sim_start_time = time.time()
                start_time = time.time()
                energy_list = []

                with torch.no_grad():
                    energy_list = node_graph.solve_global_local(handle_idx, handle_pts_tsr, simulator_config.deform_iters,
                                                                simulator_config.energy_converge_threshold, verbose=True)
                energy = node_graph.energy(handle_idx, handle_pts_tsr)
                
                deformed_pcd = node_graph.get_pcd(handle_idx=handle_idx, handle_pts_tsr=handle_pts_tsr)
                o3d.visualization.draw_geometries([rest_pcd, deformed_pcd] + arrow_lst)

                np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_debug_energy_list.npy', energy_list)
                print('optimization time:', time.time() - start_time)
                print('final energy:', energy.item())

                np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_energy.npy', energy.item())

                # plt.plot(energy_list)
                # plt.show()
                final_energy_lst.append(energy.item())

                delta_vis_pts = node_graph.get_delta_pts().detach().cpu().numpy()
                vis_pts_delta = vis_beta @ delta_vis_pts
                vis_pts_delta[:len(branch_pcd.points), :] = 0
                curr_vis_pts = all_vis_pts + vis_pts_delta
                
                all_vis_pcd = o3d.geometry.PointCloud()
                all_vis_pcd.points = o3d.utility.Vector3dVector(curr_vis_pts)
                all_vis_pcd.paint_uniform_color([0.0, 0.7, 0.7])
                o3d.io.write_point_cloud(str(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_all_vis_pcd.ply'), all_vis_pcd)

                curr_pcd = node_graph.get_pcd(handle_idx, handle_pts_tsr)
                curr_pcd.paint_uniform_color([0.0, 1.0, 0.0])
                o3d.io.write_point_cloud(str(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_curr_pts_pcd.ply'), curr_pcd)
                o3d.visualization.draw_geometries([rest_pcd, curr_pcd, all_vis_pcd] + arrow_lst)

                # o3d.io.write_point_cloud(str(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_arm_pcd.ply'), arm_pcd)

                merged_pcd = all_vis_pcd + fit_fruit_pcd #+ arm_pcd
                # o3d.visualization.draw_geometries([merged_pcd, fruit_pcd, leaf_pcd, branch_pcd, arm_pcd])
                
                vis_points, octo_points = octomap_wrapper.compute_visible_points(merged_pcd, fit_fruit_pcd, cam_center,
                                                                                 args.voxel_size, verbose=False)

                print('ray casting number visible points:', len(vis_points))
                np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_num_vis_pts.npy', len(vis_points))

                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                cam_frame.transform(cam2rob_trans)

                if len(vis_points) == 0:
                    print('INFO: no visible points')
                    continue
                else:
                    vis_pcd = o3d.geometry.PointCloud()
                    vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(vis_points))
                    vis_pcd.paint_uniform_color([0.7, 0.7, 0.0])
                    octo_pcd = o3d.geometry.PointCloud()
                    octo_pcd.points = o3d.utility.Vector3dVector(np.asarray(octo_points))
                    octo_pcd.paint_uniform_color([0.0, 0.7, 0.7])
                    o3d.visualization.draw_geometries([octo_pcd, vis_pcd, cam_frame, all_vis_pcd])

                #     o3d.io.write_point_cloud(str(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_ray_vis_pcd.ply'), vis_pcd)

                # move_end_time = time.time()
                # print('move time:', move_end_time - move_start_time)

                # # if len(vis_points) == all_vis_fit_fruit_pts_num:
                # #     print('INFO: reach maximum visible points')
                # #     break

                # if len(vis_points) >= all_vis_fit_fruit_pts_num * args.max_visible_pts_rate:
                #     print('INFO: reach maximum visible points rate:', len(vis_points) / all_vis_fit_fruit_pts_num)
                #     # break

            final_energy_lst.append(f'{grasp_id:02d}_{move_id:02d}')
            final_number_visible_pts_lst.append(f'{grasp_id:02d}_{move_id:02d}')
