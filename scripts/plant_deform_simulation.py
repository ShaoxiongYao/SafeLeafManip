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
from ssc_lmap.grasp_planner import GraspPlanner
from ssc_lmap.vis_utils import create_ball, create_arrow_lst, bool2color
from ssc_lmap.vis_utils import gen_trans_box, image_plane2pcd, create_ball

from ssc_lmap.pts_utils import trans_matrix2pose, get_largest_dbscan_component
from ssc_lmap.embed_deform_graph import prepare_sim_obj, NodeGraph
from ssc_lmap.grasp_planner import get_discrete_move_directions
from data.arm_configs import home_angles


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
    parser.add_argument('--object_boundary_threshold', type=float, default=0.015, help='object boundary threshold')    
    parser.add_argument('--num_cvx_samples', type=int, default=15, help='number of convex hull samples')    
    parser.add_argument('--pregrasp_dis', type=float, default=0.05, help='pregrasp distance')
    parser.add_argument('--sim_out_dir', type=str, default='data/sim_out', help='output directory for simulation')
    parser.add_argument('--num_box_pts', type=int, default=25)
    parser.add_argument('--deform_optimizer', type=str, default='inverse')
    parser.add_argument('--move_min_dist', type=float, default=0.01)
    parser.add_argument('--move_max_dist', type=float, default=0.05)
    parser.add_argument('--move_steps', type=int, default=3)
    parser.add_argument('--deform_iters', type=int, default=20)
    parser.add_argument('--energy_converge_threshold', type=float, default=0.01)
    parser.add_argument('--voxel_size', type=float, default=0.003)
    parser.add_argument('--close_branch_distance', type=float, default=0.05, help='close branch distance')
    parser.add_argument('--graph_voxel_size', type=float, default=0.005, help='voxel size for graph construction')
    parser.add_argument('--nn_radius', type=float, default=0.008, help='nearest neighbor radius for graph construction')
    
    parser.add_argument('--trans_params_fn', type=str, default='data/tripod_finetune_params1.json')
    
    args = parser.parse_args()
    
    time_stats_dict = {}
    
    # Load camera extrinsic parameters
    camera_params = json.load(open(args.trans_params_fn))
    cam2rob_trans = np.array(camera_params['cam2rob'])
    cam_center = cam2rob_trans[:3, 3]
    
    grasp_sample_start_time = time.time()
    

    segment_dir = Path(args.segment_dir)
    shape_complete_dir = Path(args.shape_complete_dir)
    # Load segmented point clouds
    branch_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[0]}_pcd.ply'))
    leaf_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[1]}_pcd.ply'))
    fruit_pcd = o3d.io.read_point_cloud(str(segment_dir / f'{CLASSES[2]}_pcd.ply'))
    cropped_scene_pcd = o3d.io.read_point_cloud(str(segment_dir / f'cropped_scene_pcd.ply'))
    
    # Preprocess leaf and fruit point clouds
    leaf_pcd = get_largest_dbscan_component(leaf_pcd, nn_radius=0.008, min_points=30, vis_pcd=True)
    fruit_pcd = get_largest_dbscan_component(fruit_pcd, nn_radius=0.008, min_points=30, vis_pcd=True)
    
    # Load completed triangle meshes
    fit_fruit_mesh = o3d.io.read_triangle_mesh(str(shape_complete_dir / 'completed_fruit_mesh.ply'))
    fit_fruit_pcd = fit_fruit_mesh.sample_points_poisson_disk(number_of_points=10000)

    grasp_planner = GraspPlanner(branch_pcd, object_boundary_threshold=args.object_boundary_threshold)
    grasp_frame_lst = grasp_planner.plan_grasp_leaf(leaf_pcd, num_cvx_samples=args.num_cvx_samples, 
                                                    vis_cvx_hull=False, vis_approach_pts=False, 
                                                    vis_grasp_frame=False, vis_remove_outliers=False
                                                    # vis_cvx_hull=False, vis_approach_pts=True, vis_grasp_frame=False, vis_remove_outliers=True
                                                    )
    print('number of grasp frames:', len(grasp_frame_lst))

    leaf_center = np.array(leaf_pcd.get_center())
    pregrasp_frame_lst = []
    for grasp_frame in grasp_frame_lst:
        pregrasp_frame = np.copy(grasp_frame)
        backward_direction = pregrasp_frame[:3, 3] - leaf_center
        backward_direction /= np.linalg.norm(backward_direction)
        pregrasp_frame[:3, 3] += args.pregrasp_dis * backward_direction
        pregrasp_frame_lst.append(pregrasp_frame)

    all_grasp_coord_frame_lst = []
    all_pregrasp_coord_frame_lst = []
    all_grasp_ball_lst = []
    for grasp_id, grasp_frame in enumerate(grasp_frame_lst):
        grasp_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
        pregrasp_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
        grasp_coord_frame.transform(grasp_frame)
        all_grasp_coord_frame_lst.append(grasp_coord_frame)
        pregrasp_coord_frame.transform(pregrasp_frame_lst[grasp_id])
        all_pregrasp_coord_frame_lst.append(pregrasp_coord_frame)
        all_grasp_ball_lst.append(create_ball(radius=0.01, color=[0.9, 0.3, 0.0], center=grasp_frame[:3, 3]))
    
    o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, fruit_pcd] + all_grasp_ball_lst + 
                                      all_pregrasp_coord_frame_lst + all_grasp_coord_frame_lst, **view_params)

    sim_out_dir = Path(args.sim_out_dir)
    sim_out_dir.mkdir(parents=True, exist_ok=True)
    
    move_vec_ary = get_discrete_move_directions('3D-6directions')

    final_energy_lst = []
    final_number_visible_pts_lst = []
    for grasp_id, grasp_frame in enumerate(grasp_frame_lst):

        precompute_start_time = time.time()

        box_start_pcd = gen_trans_box(0.025, args.num_box_pts, grasp_frame)
        # Prepare object points
        all_vis_pcd, all_sim_pts, handle_idx, connect_ary, edge_weights = prepare_sim_obj(box_start_pcd, branch_pcd, leaf_pcd, 
                                                                                          args.close_branch_distance, args.graph_voxel_size,
                                                                                          nn_radius=args.nn_radius, 
                                                                                          # verbose=False
                                                                                          verbose=True
                                                                                          )
        all_vis_pts = np.array(all_vis_pcd.points)

        # edge_weights = np.ones(len(connect_ary))
        node_graph = NodeGraph(all_sim_pts, connect_ary, edge_weights=edge_weights, corotate=True, device='cuda')
        rest_pcd = node_graph.get_pcd()
        rest_pcd.paint_uniform_color([0.7, 0.0, 0.7])

        print('number of nodes:', node_graph.num_pts)
        print('number of edges:', node_graph.num_edges)
        
        vis_beta = node_graph.get_pts_beta(all_vis_pts, rbf_sig=0.3, dist_max=0.05)

        line_set = node_graph.get_line_set()
        o3d.visualization.draw_geometries([line_set])

        if args.deform_optimizer == 'inverse' or args.deform_optimizer == 'inverse_plus_adam':
            # Setup handle index in graph, precompute Lapaclacian matrix and its inverse
            node_graph.set_handle_idx(handle_idx)
        
        precompute_end_time = time.time()
        print('precompute time:', precompute_end_time - precompute_start_time)
        time_stats_dict[f'grasp_{grasp_id:02d}_precompute_time'] = precompute_end_time - precompute_start_time


        for move_id, local_move_vec in enumerate(move_vec_ary):

            world_move_vec = local_move_vec.copy()

            np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_local_move_vec.npy', local_move_vec)
            print('local move vec:', local_move_vec)

            # reset the state of the graph to the initial state
            node_graph.reset_state()

            # optimizer = torch.optim.Adam(node_graph.deform_state.parameters(), lr=args.deform_lr)

            for move_step_length in np.linspace(args.move_min_dist, args.move_max_dist, num=args.move_steps):

                move_start_time = time.time()

                np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_world_move_vec.npy', 
                        move_step_length*world_move_vec)

                rest_pts_tsr = torch.tensor(all_sim_pts[handle_idx, :], dtype=torch.double, device='cuda')
                energy = node_graph.energy(handle_idx, rest_pts_tsr)
                print('initial energy:', energy.item())
                
                current_points = all_sim_pts[handle_idx].copy()
                current_points[:args.num_box_pts] += move_step_length * world_move_vec
                handle_pts_tsr = torch.tensor(current_points, dtype=torch.double, device='cuda')

                arrow_lst = create_arrow_lst(all_sim_pts[handle_idx], handle_pts_tsr.detach().cpu().numpy())
                # o3d.visualization.draw_geometries([branch_pcd, completed_branch_mesh, leaf_pcd, box_start_pcd, grasp_coord_frame] + arrow_lst)

                geom = node_graph.get_pcd(handle_idx=handle_idx, handle_pts_tsr=handle_pts_tsr)
                o3d.visualization.draw_geometries([rest_pcd, geom] + arrow_lst)

                sim_start_time = time.time()
                start_time = time.time()
                energy_list = []

                with torch.no_grad():
                    energy_list = node_graph.solve_global_local(handle_idx, handle_pts_tsr, num_iters=args.deform_iters,
                                                                energy_converge_threshold=args.energy_converge_threshold, verbose=False)
                energy = node_graph.energy(handle_idx, handle_pts_tsr)

                np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_{args.deform_optimizer}_debug_energy_list.npy', energy_list)
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
                all_vis_pcd.points = o3d.utility.Vector3dVector(curr_vis_pts)
                all_vis_pcd.paint_uniform_color([0.0, 0.7, 0.7])
                o3d.io.write_point_cloud(str(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_all_vis_pcd.ply'), all_vis_pcd)

                curr_pcd = node_graph.get_pcd(handle_idx, handle_pts_tsr)
                curr_pcd.paint_uniform_color([0.0, 1.0, 0.0])
                o3d.io.write_point_cloud(str(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_curr_pts_pcd.ply'), curr_pcd)
                # o3d.visualization.draw_geometries([rest_pcd, curr_pcd] + arrow_lst)

                # o3d.io.write_point_cloud(str(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_arm_pcd.ply'), arm_pcd)

                merged_pcd = all_vis_pcd + fit_fruit_pcd #+ arm_pcd
                # o3d.visualization.draw_geometries([merged_pcd, fruit_pcd, leaf_pcd, branch_pcd, arm_pcd])

                octomap_start_time = time.time()

                # copy a octomap scene to add fit points and arm pcd for ray tracing 
                octomap_scene_copy = octomap.OcTree(args.voxel_size)
                for pcd_pt in np.array(merged_pcd.points):
                    octo_pt_key = octomap_scene_copy.coordToKey(pcd_pt)
                    octomap_scene_copy.updateNode(octo_pt_key, octomap_scene_copy.getProbHitLog(), True)
                octomap_scene_copy.updateInnerOccupancy()
                
                # Ray tracing to check if the points are visible
                start_time = time.time()
                vis_dict= dict()
                octo_pts = []
                for pcd_pt in np.array(fit_fruit_pcd.points):
                    octo_pt_key = octomap_scene_copy.coordToKey(pcd_pt)
                    if tuple([octo_pt_key[0], octo_pt_key[1], octo_pt_key[2]]) in vis_dict:
                        continue
                    octo_pt = octomap_scene_copy.keyToCoord(octo_pt_key)
                    octo_pts.append(octo_pt)
                    # do ray tracing to check if the point is visible
                    origin = octomap_scene_copy.keyToCoord(octomap_scene_copy.coordToKey(cam_center))
                    direction = octo_pt - origin
                    direction /= np.linalg.norm(direction)
                    end_pt = origin
                    hit = octomap_scene_copy.castRay(origin, direction, end_pt, True, 1.5) 
                    if hit:
                        end_pt_key = octomap_scene_copy.coordToKey(end_pt)
                        if end_pt_key == octo_pt_key:
                            octo_pt_tuple = tuple([octo_pt_key[0], octo_pt_key[1], octo_pt_key[2]])
                            vis_dict[octo_pt_tuple] = 1
                print('ray tracing time:', time.time() - start_time)

                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                cam_frame.transform(cam2rob_trans)

                vis_points = []
                for k in vis_dict.keys():
                    octo_k = octomap.OcTreeKey()
                    octo_k[0] = k[0]
                    octo_k[1] = k[1]
                    octo_k[2] = k[2]
                    vis_points.append(octomap_scene_copy.keyToCoord(octo_k))
                vis_points = np.array(vis_points)
                print('ray casting number visible points:', len(vis_points))
                np.save(sim_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_num_vis_pts.npy', len(vis_points))

                # final_number_visible_pts_lst.append(len(vis_points))

                # if f'grasp_{grasp_id:02d}_move_{move_id:02d}_octomap_time' not in time_stats_dict:
                #     time_stats_dict[f'grasp_{grasp_id:02d}_move_{move_id:02d}_octomap_time'] = 0.0
                # else:
                #     time_stats_dict[f'grasp_{grasp_id:02d}_move_{move_id:02d}_octomap_time'] += time.time() - octomap_start_time

                if len(vis_points) == 0:
                    print('INFO: no visible points')
                    continue
                else:
                    vis_pcd = o3d.geometry.PointCloud()
                    vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(vis_points))
                    vis_pcd.paint_uniform_color([0.7, 0.7, 0.0])
                    octo_pcd = o3d.geometry.PointCloud()
                    octo_pcd.points = o3d.utility.Vector3dVector(np.asarray(octo_pts))
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
