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
from ssc_lmap.scene_consistent_deepsdf import SceneConsistentDeepSDF
from ssc_lmap.realsense_utils import load_intrinsics_from_json, save_intrinsics_from_json
from ssc_lmap.embed_deform_graph import prepare_sim_obj
from ssc_lmap.pts_utils import trans_matrix2pose

import tqdm
import json
from copy import deepcopy


import context
from ssc_lmap.pts_utils import is_free_frontier_point, sdf_negetive_inside_trimesh
from ssc_lmap.vis_utils import create_arrow_lst, gen_trans_box, bool2color, image_plane2pcd, create_ball
from ssc_lmap.branch_completion import BranchCompletion
from ssc_lmap.grasp_planner import GraspPlanner
from data.arm_configs import home_angles, qlimits, home_pose

view_params = {	
    "front" : [ -0.72121813879241847, -0.67127630598545718, 0.17097519498254365 ],
    "lookat" : [ -0.19298926386875953, 0.74899148655268699, 0.44262800215821724 ],
    "up" : [ 0.036966100337826095, 0.20917298400491585, 0.97717970209593119 ],
    "zoom" : 0.27999999999999958
}

view_params = {	
    "front" : [ -0.72719746380345296, -0.65442554290856003, 0.20714984293178301 ],
    "lookat" : [ -0.16539800516476486, 0.737540449204104, 0.41621635030538628 ],
    "up" : [ 0.10859730785732971, 0.18829519911886816, 0.97608992552680629 ],
    "zoom" : 0.21999999999999958
}

color_intr, depth_intr = load_intrinsics_from_json('data/camera_intrinsics.json')

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
    
    if args.add_real_ur5:
        IK_SOLVER_WORKING = False
        while not IK_SOLVER_WORKING:
            ####### this is due to TCP linking in the UR5Real class which makes the IK no solution sometimes #######
            random_sleep_time = np.random.uniform(0, 10) / 10.0
            print(f'Random sleep time: {random_sleep_time}')
            time.sleep(random_sleep_time)
            
            arm_real = UR5Real(ur5_ip=args.ur5_ip)
            arm_real.set_default_gripper_tcp()
            arm_real.set_default_gripper_payload()

            home_pose_shift = home_pose.copy()
            home_pose_shift[0] += 0.005
            home_pose_shift[1] += 0.005
            home_pose_shift[2] += 0.005
            arm_angles = arm_real.solve_ik(home_pose_shift, qnear=home_angles, qlimits=qlimits)
            print ('arm_angles:', [np.rad2deg(x) for x in arm_angles])
            if len(arm_angles) != 0:
                print('IK solver working')
                IK_SOLVER_WORKING = True
            else:
                print('IK solver not working, retrying')
                del arm_real

    # Load camera extrinsic parameters
    camera_params = json.load(open(args.trans_params_fn))
    cam2rob_trans = np.array(camera_params['cam2rob'])
    rob2cam_trans = np.array(camera_params['rob2cam'])
    cam_center = cam2rob_trans[:3, 3]

    arm_sim = UR5Sim('data/robots/ur5_with_robotiq85.rob', num_pcd_pts=5000)
    arm_sim.close_gripper()
    arm_sim.set_angles(home_angles)
    home_arm_pcd = arm_sim.get_arm_pcd()

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

    image = cv2.imread(str(obj_data_dir / 'original_image.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pcd = image_plane2pcd(image, color_intr.width, color_intr.height, z=0)

    
    time_free_space_start = time.time()
    #selecte raw_scene_pcd points close to the fruit
    fruit_center = np.mean(fruit_pts, axis=0)
    raw_scene_pcd = raw_scene_pcd.voxel_down_sample(voxel_size=args.voxel_size)
    #create an OctoMap
    free_octomap_scene = octomap.OcTree(args.voxel_size)
    #compute free space are close to object
    coord_range = int(args.fruit_max_size / args.voxel_size * args.free_space_bbx_scale)
    #initialize unknown space
    for x in range(-coord_range, coord_range):
        for y in range(-coord_range, coord_range):
            for z in range(-coord_range, coord_range):
                pts = fruit_center + np.array([x*args.voxel_size, y*args.voxel_size, z*args.voxel_size])
                key = free_octomap_scene.coordToKey(np.array([pts[0], pts[1], pts[2]]))
                free_octomap_scene.updateNode(key, update=0.0, lazy_eval=True) # unknown space log odds is 0.0
    free_octomap_scene.updateInnerOccupancy()
    # insert the raw_scene_pcd into the octomap
    free_octomap_scene.insertPointCloud(np.array(raw_scene_pcd.points), cam_center)
    free_octomap_scene.updateInnerOccupancy()

    # compute the free space points
    free_space_lst = []
    for x in range(-coord_range, coord_range):
        for y in range(-coord_range, coord_range):
            for z in range(-coord_range, coord_range):
                pts = fruit_center + np.array([x*args.voxel_size, y*args.voxel_size, z*args.voxel_size])
                if np.linalg.norm(pts - fruit_center) > 2.0 * args.fruit_max_size * args.free_space_bbx_scale:
                    continue
                if is_free_frontier_point(pts, free_octomap_scene, voxel_size = args.voxel_size, layer = args.free_space_layer):
                    free_space_lst.append(pts)
    free_space_pcd = o3d.geometry.PointCloud()
    free_space_pcd.points = o3d.utility.Vector3dVector(np.array(free_space_lst))
    print ('free space pcd created')
    time_free_space_end = time.time()
    print ('free space time:', time_free_space_end - time_free_space_start)
    time_stats_dict['init_free_space_time'] = time_free_space_end - time_free_space_start

    o3d.visualization.draw_geometries([free_space_pcd, fruit_pcd, branch_pcd, leaf_pcd])
    o3d.io.write_point_cloud(str(obj_data_dir / 'free_space_pcd.ply'), free_space_pcd)

    # load deep sdf decoder and init latent code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DeepSDF_DIR = '/home/user/pan/embed_graph_push_planning/ssc_lmap/HortiMapping/deepsdf/models/sweetpepper_32'
    DeepSDF_DIR = 'ssc_lmap/HortiMapping/deepsdf/models/sweetpepper_32'
    checkpoint = "latest"
    semantic_deepsdf = SceneConsistentDeepSDF(DeepSDF_DIR)

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
    fit_fruit_pcd = fit_fruit_mesh.sample_points_poisson_disk(number_of_points=10000)

    time_stats_dict['fit_time'] = out_dict['time']

    o3d.io.write_triangle_mesh(str(obj_data_dir / 'fit_fruit_mesh.ply'), fit_fruit_mesh)
    o3d.io.write_point_cloud(str(obj_data_dir / 'fit_fruit_pcd.ply'), fit_fruit_pcd)
    with open(obj_data_dir / 'fit_params.json', 'w') as f:
        fit_params = {
            'translation': out_dict['translation'].tolist(),
            'rotation': out_dict['rotation'].tolist(),
            'scale': out_dict['scale'].tolist()   
        }
        json.dump(fit_params, f, indent=2)
    torch.save(out_dict['latent'], f'{obj_data_dir}/fit_latent.pth')
    
    start_time = time.time()
    fit_fruit_pcd_img_frame = deepcopy(fit_fruit_pcd)
    fit_fruit_pcd_img_frame.transform(rob2cam_trans)
    pixel_lst = []
    i_lst, j_lst = [], []
    for pt in np.array(fit_fruit_pcd_img_frame.points):
        color_pixel_pojected = rs.rs2_project_point_to_pixel(color_intr, pt)
        pixel_lst.append(color_pixel_pojected)
        if color_pixel_pojected[0] < 0 or color_pixel_pojected[0] >= color_intr.width \
            or color_pixel_pojected[1] < 0 or color_pixel_pojected[1] >= color_intr.height:
            continue
        mask_val_lst = mask_array[:, int(color_pixel_pojected[1]), int(color_pixel_pojected[0])]
        i_lst.append(int(color_pixel_pojected[1]))
        j_lst.append(int(color_pixel_pojected[0]))
    i_min, i_max = min(i_lst), max(i_lst)
    j_min, j_max = min(j_lst), max(j_lst)
    print('projection time:', time.time() - start_time)
        
    scene_pcd = leaf_pcd + branch_pcd + fruit_pcd

    # Using octomap to repersent scene and detect visible points afterwards (update once in system loop)
    start_time = time.time()
    octomap_scene = octomap.OcTree(args.voxel_size)
    for pcd_pt in np.array(fruit_pcd.points):
        octo_pt_key = octomap_scene.coordToKey(pcd_pt)
        octomap_scene.updateNode(octo_pt_key, octomap_scene.getProbHitLog(), lazy_eval=True) 
    octomap_scene.updateInnerOccupancy()
    vis_check_scene_pcd = leaf_pcd + branch_pcd ##### leaf + branch are occluding fruit
    for pcd_pt in np.array(vis_check_scene_pcd.points):
        octo_pt_key = octomap_scene.coordToKey(pcd_pt)
        octomap_scene.updateNode(octo_pt_key, octomap_scene.getProbHitLog(), lazy_eval=True) 
    octomap_scene.updateInnerOccupancy()
    print('octomap time:', time.time() - start_time)

    all_fruit_pts_lst = []
    vis_fruit_pts_lst = []
    fruit_dict= dict()
    for pcd_pt in np.array(fruit_pcd.points):
        octo_pt_key = octomap_scene.coordToKey(pcd_pt)
        if tuple([octo_pt_key[0], octo_pt_key[1], octo_pt_key[2]]) in fruit_dict:
            continue
        octo_pt = octomap_scene.keyToCoord(octo_pt_key)
        all_fruit_pts_lst.append(octo_pt)
        # do ray tracing to check if the point is visible
        origin = octomap_scene.keyToCoord(octomap_scene.coordToKey(cam_center))
        direction = octo_pt - origin
        direction /= np.linalg.norm(direction)
        end_pt = origin
        hit = octomap_scene.castRay(origin, direction, end_pt, True, 1.5) 
        if hit:
            end_pt_key = octomap_scene.coordToKey(end_pt)
            if end_pt_key == octo_pt_key:
                vis_fruit_pts_lst.append(octo_pt)
                octo_pt_tuple = tuple([octo_pt_key[0], octo_pt_key[1], octo_pt_key[2]])
                fruit_dict[octo_pt_tuple] = 1
    all_vis_fruit_pts_num = len(fruit_dict)
    print('number of vis fruit points:', all_vis_fruit_pts_num)
    # vis_fruit_pcd = o3d.geometry.PointCloud()
    # vis_fruit_pcd.points = o3d.utility.Vector3dVector(np.asarray(vis_fruit_pts_lst))
    # vis_fruit_pcd.paint_uniform_color([0.7, 0.7, 0.0])
    # all_fruit_pcd = o3d.geometry.PointCloud()
    # all_fruit_pcd.points = o3d.utility.Vector3dVector(np.asarray(all_fruit_pts_lst))
    # all_fruit_pcd.paint_uniform_color([0.7, 0.0, 1.0])
    # cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # cam_frame.transform(cam2rob_trans)
    # o3d.visualization.draw_geometries([all_fruit_pcd, vis_fruit_pcd, cam_frame, vis_check_scene_pcd])

    # Using octomap to repersent scene and detect visible points afterwards (update once in system loop)
    start_time = time.time()
    octomap_scene = octomap.OcTree(args.voxel_size)
    for pcd_pt in np.array(fit_fruit_pcd.points):
        octo_pt_key = octomap_scene.coordToKey(pcd_pt)
        octomap_scene.updateNode(octo_pt_key, octomap_scene.getProbHitLog(), lazy_eval=True) 
    octomap_scene.updateInnerOccupancy()
    print('octomap time:', time.time() - start_time)
    all_fruit_pts_lst = []
    vis_fruit_pts_lst = []
    fruit_dict= dict()
    for pcd_pt in np.array(fit_fruit_pcd.points):
        octo_pt_key = octomap_scene.coordToKey(pcd_pt)
        if tuple([octo_pt_key[0], octo_pt_key[1], octo_pt_key[2]]) in fruit_dict:
            continue
        octo_pt = octomap_scene.keyToCoord(octo_pt_key)
        all_fruit_pts_lst.append(octo_pt)
        # do ray tracing to check if the point is visible
        origin = octomap_scene.keyToCoord(octomap_scene.coordToKey(cam_center))
        direction = octo_pt - origin
        direction /= np.linalg.norm(direction)
        end_pt = origin
        hit = octomap_scene.castRay(origin, direction, end_pt, True, 1.5) 
        if hit:
            end_pt_key = octomap_scene.coordToKey(end_pt)
            if end_pt_key == octo_pt_key:
                vis_fruit_pts_lst.append(octo_pt)
                octo_pt_tuple = tuple([octo_pt_key[0], octo_pt_key[1], octo_pt_key[2]])
                fruit_dict[octo_pt_tuple] = 1
    all_vis_fit_fruit_pts_num = len(fruit_dict)
    print('number of vis fit fruit points:', all_vis_fit_fruit_pts_num)

    grasp_sample_start_time = time.time()
    if args.sample_action_mode == '3D-6directions':
        # Compute 6 move directions in the x, y, z space
        move_vec_ary = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
    elif args.sample_action_mode == '3D-14directions':
        # Compute 6 move directions in the x, y, z space
        move_vec_ary = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
        # Compute 8 move directions in the eight corners of the unit cube, which should be normalized
        move_vec_ary = np.concatenate([move_vec_ary, 
                                       np.array([[ 1, 1, 1], [ 1, 1, -1], [ 1, -1, 1], [ 1, -1, -1],
                                                 [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]) / np.sqrt(3)],
                                       axis=0)
    elif args.sample_action_mode == '2D-8directions':
        # Compute 8 move directions in the x, z plane
        angles_lst = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
        move_vec_ary = np.array([[0, np.cos(angle), np.sin(angle)] for angle in angles_lst])

    grasp_planner = GraspPlanner(branch_pcd, object_boundary_threshold=args.object_boundary_threshold)
    grasp_frame_lst = grasp_planner.plan_grasp_leaf(leaf_pcd, num_cvx_samples=args.num_cvx_samples, 
                                                    vis_cvx_hull=False, vis_approach_pts=False, vis_grasp_frame=False, vis_remove_outliers=False
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
    
    if args.add_real_ur5:
        arm_sim.set_angles(arm_real.get_angles())
        arm_pcd = arm_sim.get_arm_pcd()
    
    time_stats_dict['grasp_sample_time'] = time.time() - grasp_sample_start_time

    o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, fruit_pcd] + all_grasp_ball_lst + all_pregrasp_coord_frame_lst + all_grasp_coord_frame_lst, **view_params)

    plan_move_out_dir = Path(args.out_data_dir) / f'plan_move_{args.obj_name}_{args.exp_id}'
    plan_move_out_dir.mkdir(parents=True, exist_ok=True)

    o3d.io.write_triangle_mesh(str(plan_move_out_dir / f'fit_fruit_mesh.ply'), fit_fruit_mesh)
    o3d.io.write_point_cloud(str(plan_move_out_dir / f'fit_fruit_pcd.ply'), fit_fruit_pcd)
    with open(plan_move_out_dir / 'fit_params.json', 'w') as f:
        fit_params = {
            'translation': out_dict['translation'].tolist(),
            'rotation': out_dict['rotation'].tolist(),
            'scale': out_dict['scale'].tolist(),
            'img_i_min': i_min, 'img_i_max': i_max,
            'img_j_min': j_min, 'img_j_max': j_max,           
        }
        json.dump(fit_params, f, indent=2)
    torch.save(out_dict['latent'], f'{plan_move_out_dir}/fit_latent.pth')

    # Save action sampling information to file
    sample_action_info = {
        'sample_action_mode': args.sample_action_mode,
        'num_move_vec': len(move_vec_ary),
        'num_fruit_pts': all_vis_fruit_pts_num,
        'num_fit_fruit_pts': all_vis_fit_fruit_pts_num,
        'trans_params_fn': args.trans_params_fn
    }
    with open(plan_move_out_dir / 'sample_action_info.json', 'w') as f:
        json.dump(sample_action_info, f, indent=2)
    
    np.save(plan_move_out_dir / 'move_vec_ary.npy', move_vec_ary)

    with open(plan_move_out_dir / 'camera_extrinsic.json', 'w') as f:
        json.dump(camera_params, f, indent=2)
    
    save_intrinsics_from_json(str(plan_move_out_dir / 'camera_intrinsics.json'), color_intr, depth_intr)

    final_energy_lst = []
    final_number_visible_pts_lst = []
    for grasp_id, grasp_frame in enumerate(grasp_frame_lst):
        if grasp_id != 4:
            continue

        if args.add_real_ur5:
            ##### pregrasp point #####
            np.save(plan_move_out_dir / f'pregrasp_{grasp_id:02d}_frame.npy', pregrasp_frame_lst[grasp_id])
            pregrasp_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            pregrasp_coord_frame.transform(pregrasp_frame_lst[grasp_id])
            pregrasp_pose = trans_matrix2pose(pregrasp_frame_lst[grasp_id])

            reachable = True
            if args.add_real_ur5:
                pregrasp_arm_angles = arm_real.solve_ik(pregrasp_pose, qnear=home_angles, qlimits=qlimits)
                print('pregrasp_arm_angles:', [np.rad2deg(x) for x in pregrasp_arm_angles])
                if len(pregrasp_arm_angles) == 0:
                    reachable = False
                np.save(plan_move_out_dir / f'grasp_{grasp_id:02d}_pregrasp_arm_angles.npy', pregrasp_arm_angles)
                arm_sim.set_angles(pregrasp_arm_angles)
            if not reachable:
                print('INFO: pregrasp is not reachable', pregrasp_pose)
                continue

            arm_pcd = arm_sim.get_gripper_open2close_pcd()
            start_arm2fruit_closeform = semantic_deepsdf.points_sdf_value(np.array(arm_pcd.points))
            # plt.hist(start_arm2fruit_closeform, bins=100)
            # plt.show()

            colors = np.zeros((len(arm_pcd.points), 3))
            colors[start_arm2fruit_closeform < 0] = [1, 0, 0]
            colors[start_arm2fruit_closeform >= 0] = [0, 1, 0]
            arm_pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, fruit_pcd, fit_fruit_pcd, arm_pcd, pregrasp_coord_frame])
        else:
            pregrasp_arm_angles = home_angles
            pregrasp_pose = home_pose

        ##### grasp point #####

        np.save(plan_move_out_dir / f'grasp_{grasp_id:02d}_frame.npy', grasp_frame)

        grasp_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        grasp_coord_frame.transform(grasp_frame)
        grasp_pose = trans_matrix2pose(grasp_frame)

        world_move_vec_lst = []
        for move_id, local_move_vec in enumerate(move_vec_ary):
            world_move_vec = np.dot(grasp_frame[:3, :3], local_move_vec)
            world_move_vec_lst.append(world_move_vec)
        
        # arrow_lst = create_arrow_lst(np.array([grasp_pose[:3]]*len(move_vec_ary)), 
        #                              np.array(grasp_pose[:3]) + 0.05*np.array(world_move_vec_lst), scale=2)
        # o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, fruit_pcd] + arrow_lst, **view_params)

        reachable = True
        if args.add_real_ur5:
            start_arm_angles = arm_real.solve_ik(grasp_pose, qnear=pregrasp_arm_angles, qlimits=qlimits)
            print('start_arm_angles:', [np.rad2deg(x) for x in start_arm_angles])
            if len(start_arm_angles) == 0:
                reachable = False
            np.save(plan_move_out_dir / f'grasp_{grasp_id:02d}_start_arm_angles.npy', start_arm_angles)
            arm_sim.set_angles(start_arm_angles)
        if not reachable:
            print('INFO: grasp is not reachable', grasp_pose)
            continue

        # arm_sim.set_angles(start_arm_angles)
        # arm_mesh_lst = arm_sim.get_mesh_lst()
        # arm_box_pcd = arm_sim.get_box_pcd()
        # o3d.visualization.draw_geometries([arm_box_pcd, leaf_pcd, branch_pcd, fruit_pcd, fit_fruit_pcd, grasp_coord_frame] + arm_mesh_lst)
        
        ### TEST GRASPING POINTS ONLY###
        # continue
        
        collision_start_time = time.time()

        arm_pcd = arm_sim.get_gripper_open2close_pcd()
        start_arm2fruit_closeform = semantic_deepsdf.points_sdf_value(np.array(arm_pcd.points))
        # plt.hist(start_arm2fruit_closeform, bins=100)
        # plt.show()

        colors = np.zeros((len(arm_pcd.points), 3))
        colors[start_arm2fruit_closeform < 0] = [1, 0, 0]
        colors[start_arm2fruit_closeform >= 0] = [0, 1, 0]
        arm_pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, completed_branch_mesh, fit_fruit_pcd, arm_pcd, home_arm_pcd])

        # box_pcd = arm_sim.get_box_pcd()
        # arm_pcd = arm_sim.get_arm_pcd()
        # o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, completed_branch_mesh, fruit_pcd, fit_fruit_pcd, arm_pcd, box_pcd, grasp_coord_frame] + arrow_lst)

        arm2branch_distance = arm_pcd.compute_point_cloud_distance(branch_pcd)
        arm2branch_idx = np.where(np.asarray(arm2branch_distance) < args.collision_distance * 1.5)[0]
        if arm2branch_idx.size > 0:
            arm2branch_pcd = arm_pcd.select_by_index(arm2branch_idx)
        else:
            arm2branch_pcd = o3d.geometry.PointCloud()
        start_arm2branch_sdf = sdf_negetive_inside_trimesh(simplified_completed_branch_mesh.vertices, simplified_completed_branch_mesh.triangles, np.array(arm2branch_pcd.points))

        if f'grasp_{grasp_id:02d}_collision_time' not in time_stats_dict:
            time_stats_dict[f'grasp_{grasp_id:02d}_collision_time'] = 0.0
        else:
            time_stats_dict[f'grasp_{grasp_id:02d}_collision_time'] += time.time() - collision_start_time
        print ('collision time:', time.time() - collision_start_time)
        
        if np.sum(start_arm2fruit_closeform < args.collision_distance) >= 1:
            arm_pcd.colors = o3d.utility.Vector3dVector(bool2color(start_arm2fruit_closeform < args.collision_distance, [0, 0, 1], [1, 0, 0]))
            # o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, completed_branch_mesh, fruit_pcd, fit_fruit_pcd, arm_pcd, grasp_coord_frame] + arrow_lst)
            print('number of points inside superellipsoid:', np.sum(start_arm2fruit_closeform < 0))
            print('number of points unsafe:', np.sum(start_arm2fruit_closeform < args.collision_distance))
            print('INFO: start config is too close to the fruit, skip the action')
            continue

        if np.sum(start_arm2branch_sdf < args.collision_distance) >= 1:
            arm_pcd.colors = o3d.utility.Vector3dVector(bool2color(start_arm2branch_sdf < args.collision_distance, [0, 0, 1], [1, 0, 0]))
            # o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, completed_branch_mesh, fruit_pcd, fit_fruit_pcd, arm_pcd, grasp_coord_frame] + arrow_lst)
            print('number of points inside branch:', np.sum(start_arm2branch_sdf < 0))
            print('number of points unsafe:', np.sum(start_arm2branch_sdf < args.collision_distance))
            print('INFO: start config is too close to the branch, skip the action')
            continue
            
        precompute_start_time = time.time()

        box_start_pcd = gen_trans_box(0.025, args.num_box_pts, grasp_frame)
        # Prepare object points
        all_vis_pcd, all_sim_pts, handle_idx, connect_ary, edge_weights = prepare_sim_obj(box_start_pcd, branch_pcd, leaf_pcd, 
                                                                                          args.close_branch_distance, args.graph_voxel_size,
                                                                                          nn_radius=args.nn_radius, 
                                                                                        #   verbose=False
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

        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
        # coord_frame.transform(grasp_frame)
        line_set = node_graph.get_line_set()
        o3d.visualization.draw_geometries([line_set])

        if args.deform_optimizer == 'inverse' or args.deform_optimizer == 'inverse_plus_adam':
            # Setup handle index in graph, precompute Lapaclacian matrix and its inverse
            node_graph.set_handle_idx(handle_idx)
        
        precompute_end_time = time.time()
        print('precompute time:', precompute_end_time - precompute_start_time)
        time_stats_dict[f'grasp_{grasp_id:02d}_precompute_time'] = precompute_end_time - precompute_start_time

        ###### DEBUG ONLY ######
        # if not (grasp_id == 7):
        #     continue
        # arrow_lst = []
        # for move_id, local_move_vec in enumerate(move_vec_ary):
        #     # world_move_vec = np.dot(grasp_frame[:3, :3], local_move_vec)
        #     world_move_vec = local_move_vec.copy()
        #     if world_move_vec[2] < 0:
        #         pass
        #     else:
        #         grasp_point = grasp_frame[:3, 3]
        #         grasp_point_pcd = o3d.geometry.PointCloud()
        #         grasp_point_pcd.points = o3d.utility.Vector3dVector([grasp_point])
        #         grasp2branch_distance = np.array(completed_branch_pcd.compute_point_cloud_distance(grasp_point_pcd))
        #         cloest_grasp2branch_pt = np.array(completed_branch_pcd.points)[np.argmin(grasp2branch_distance), :]
        #         grasp2branch_vec = cloest_grasp2branch_pt - grasp_point
        #         grasp2branch_vec /= np.linalg.norm(grasp2branch_vec)
        #         move2branch_theta = np.arccos(np.dot(grasp2branch_vec, world_move_vec))
        #         if move2branch_theta > np.deg2rad(args.move2branch_theta_limit):
        #             pass
        #         else:
        #             arrow = create_arrow_lst(np.array([grasp_frame[:3, 3]]), np.array([grasp_frame[:3, 3] + 0.1*world_move_vec]))
        #             arrow_lst += arrow
        # o3d.visualization.draw_geometries([rest_pcd, grasp_coord_frame] + arrow_lst)

        for move_id, local_move_vec in enumerate(move_vec_ary):
            ### Use the gravity direction as the piror knowledge to filter out the move direction
            if move_id != 4:
                continue

            # world_move_vec = np.dot(grasp_frame[:3, :3], local_move_vec)
            world_move_vec = local_move_vec.copy()
            # if world_move_vec[2] < 0:
            #     print ('skip: move direction:', world_move_vec)
            #     continue

            ### Use the branch direction as the piror knowledge to filter out the move direction
            grasp_point = grasp_frame[:3, 3]
            grasp_point_pcd = o3d.geometry.PointCloud()
            grasp_point_pcd.points = o3d.utility.Vector3dVector([grasp_point])
            grasp2branch_distance = np.array(completed_branch_pcd.compute_point_cloud_distance(grasp_point_pcd))
            cloest_grasp2branch_pt = np.array(completed_branch_pcd.points)[np.argmin(grasp2branch_distance), :]
            grasp2branch_vec = cloest_grasp2branch_pt - grasp_point
            grasp2branch_vec /= np.linalg.norm(grasp2branch_vec)
            move2branch_theta = np.arccos(np.dot(grasp2branch_vec, world_move_vec))
            if move2branch_theta > np.deg2rad(args.move2branch_theta_limit):
                # pts1 = np.array([grasp_point, grasp_point])
                # pts2 = np.array([cloest_grasp2branch_pt, grasp_point + 0.1*world_move_vec])
                # arrow_lst = create_arrow_lst(pts1, pts2)
                # o3d.visualization.draw_geometries([branch_pcd, completed_branch_mesh, leaf_pcd, fruit_pcd] + arrow_lst)
                print('skip: move2branch_theta:', np.rad2deg(move2branch_theta))
                continue

            np.save(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_local_move_vec.npy', local_move_vec)
            print('local move vec:', local_move_vec)

            # reset the state of the graph to the initial state
            node_graph.reset_state()

            optimizer = torch.optim.Adam(node_graph.deform_state.parameters(), lr=args.deform_lr)

            for move_step_length in np.linspace(args.move_min_dist, args.move_max_dist, num=args.move_steps):

                move_start_time = time.time()

                np.save(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_world_move_vec.npy', 
                        move_step_length*world_move_vec)

                rest_pts_tsr = torch.tensor(all_sim_pts[handle_idx, :], dtype=torch.double, device='cuda')
                energy = node_graph.energy(handle_idx, rest_pts_tsr)
                print('initial energy:', energy.item())
                
                current_points = all_sim_pts[handle_idx].copy()
                current_points[:args.num_box_pts] += move_step_length * world_move_vec
                handle_pts_tsr = torch.tensor(current_points, dtype=torch.double, device='cuda')

                arrow_lst = create_arrow_lst(all_sim_pts[handle_idx], handle_pts_tsr.detach().cpu().numpy())
                # o3d.visualization.draw_geometries([branch_pcd, completed_branch_mesh, leaf_pcd, box_start_pcd, grasp_coord_frame] + arrow_lst)

                # geom = node_graph.get_pcd(handle_idx=handle_idx, handle_pts_tsr=handle_pts_tsr)
                # o3d.visualization.draw_geometries([rest_pcd, geom] + arrow_lst)

                sim_start_time = time.time()
                start_time = time.time()
                energy_list = []
                if args.deform_optimizer == 'adam':
                    ######
                    #--deform_optimizer adam --deform_iters 3000 --deform_lr 0.001 --energy_converge_threshold 0.00001 --energy_converge_window 10\
                    ######
                    for iter in range(args.deform_iters):
                        # start_time = time.time()
                        energy = node_graph.energy(handle_idx, handle_pts_tsr)
                        optimizer.zero_grad()
                        energy.backward()
                        optimizer.step()
                        print('energy:', energy.item())
                        energy_list.append(energy.item())
                        if iter > args.energy_converge_window + 2:
                            coverage = True
                            for energy_last_idx in range(1, args.energy_converge_window + 1):
                                if np.abs(energy_list[-energy_last_idx] - energy_list[-energy_last_idx-1]) > args.energy_converge_threshold:
                                    coverage = False
                                    break
                            if coverage:
                                print ('converged with threshold:', args.energy_converge_threshold)
                                print ('converged with window:', args.energy_converge_window)
                                break
                        if iter == args.deform_iters - 1:
                            print ('not converged stop with maximum iterations')

                elif args.deform_optimizer == 'inverse':
                    ######
                    #--deform_optimizer inverse --deform_iters 100 --energy_converge_threshold 0.00001\
                    ######
                    with torch.no_grad():
                        energy_list = node_graph.solve_global_local(handle_idx, handle_pts_tsr, num_iters=args.deform_iters,
                                                                    energy_converge_threshold=args.energy_converge_threshold, verbose=False)
                    energy = node_graph.energy(handle_idx, handle_pts_tsr)

                elif args.deform_optimizer == 'inverse_plus_adam':
                    ######
                    #--deform_optimizer inverse_plus_adam --deform_iters 3000 --deform_lr 0.001 --energy_converge_threshold 0.00001 --energy_converge_window 10\
                    ######
                    with torch.no_grad():
                        energy_list = node_graph.solve_global_local(handle_idx, handle_pts_tsr, num_iters=10, verbose=False)
                    energy = node_graph.energy(handle_idx, handle_pts_tsr)
                    for iter in range(args.deform_iters):
                        # start_time = time.time()
                        energy = node_graph.energy(handle_idx, handle_pts_tsr)
                        optimizer.zero_grad()
                        energy.backward()
                        optimizer.step()
                        print('energy:', energy.item())
                        energy_list.append(energy.item())
                        if iter > args.energy_converge_window + 2:
                            coverage = True
                            for energy_last_idx in range(1, args.energy_converge_window + 1):
                                if np.abs(energy_list[-energy_last_idx] - energy_list[-energy_last_idx-1]) > args.energy_converge_threshold:
                                    coverage = False
                                    break
                            if coverage:
                                print ('converged with threshold:', args.energy_converge_threshold)
                                print ('converged with window:', args.energy_converge_window)
                                break
                    if iter == args.deform_iters - 1:
                        print ('not converged stop with maximum iterations')

                np.save(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_{args.deform_optimizer}_debug_energy_list.npy', energy_list)
                print('optimization time:', time.time() - start_time)
                print('final energy:', energy.item())

                np.save(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_energy.npy', energy.item())

                # plt.plot(energy_list)
                # plt.show()
                final_energy_lst.append(energy.item())

                delta_vis_pts = node_graph.get_delta_pts().detach().cpu().numpy()
                vis_pts_delta = vis_beta @ delta_vis_pts
                vis_pts_delta[:len(branch_pcd.points), :] = 0
                curr_vis_pts = all_vis_pts + vis_pts_delta
                all_vis_pcd.points = o3d.utility.Vector3dVector(curr_vis_pts)
                all_vis_pcd.paint_uniform_color([0.0, 0.7, 0.7])
                o3d.io.write_point_cloud(str(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_all_vis_pcd.ply'), all_vis_pcd)

                curr_pcd = node_graph.get_pcd(handle_idx, handle_pts_tsr)
                curr_pcd.paint_uniform_color([0.0, 1.0, 0.0])
                o3d.io.write_point_cloud(str(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_curr_pts_pcd.ply'), curr_pcd)
                # o3d.visualization.draw_geometries([rest_pcd, curr_pcd] + arrow_lst)        

                if f'grasp_{grasp_id:02d}_move_{move_id:02d}_sim_time' not in time_stats_dict:
                    time_stats_dict[f'grasp_{grasp_id:02d}_move_{move_id:02d}_sim_time'] = 0.0
                else:
                    time_stats_dict[f'grasp_{grasp_id:02d}_move_{move_id:02d}_sim_time'] += time.time() - sim_start_time

                reachable = True
                if args.add_real_ur5:
                    new_grasp_pose = grasp_pose.copy()
                    new_grasp_pose[:3] += move_step_length * world_move_vec
                    arm_angles = arm_real.solve_ik(new_grasp_pose, qnear=start_arm_angles, qlimits=qlimits)
                    if len(arm_angles) == 0:
                        reachable = False
                    print('grasp:', grasp_id, 'move:', move_id, 'move_length:', move_step_length)
                    arm_sim.set_angles(arm_angles)
                    np.save(plan_move_out_dir / f'grasp_{grasp_id:02d}_move_{move_id:02d}_arm_angles.npy', arm_angles)
                if not reachable:
                    print('INFO: final grasp is not reachable', new_grasp_pose)
                    break
                    
                o3d.io.write_point_cloud(f'out_data/vis_sim_process/pull_artificial_100/grasp_{grasp_id:02d}_move_{move_id:02d}_{move_step_length:.05f}.ply', all_vis_pcd)
                # arm_sim.set_angles(arm_angles)
                arm_mesh_lst = arm_sim.get_mesh_lst()
                arm_mesh = sum(arm_mesh_lst, o3d.geometry.TriangleMesh())
                o3d.io.write_triangle_mesh(f'out_data/vis_sim_process/pull_artificial_100/grasp_{grasp_id:02d}_move_{move_id:02d}_{move_step_length:.05f}_arm.ply', arm_mesh)
                o3d.io.write_triangle_mesh(f'out_data/vis_sim_process/pull_artificial_100/completed_branch_mesh.ply', completed_branch_mesh)
                o3d.io.write_triangle_mesh(f'out_data/vis_sim_process/pull_artificial_100/fit_fruit_mesh.ply', fit_fruit_mesh)
                # o3d.visualization.draw_geometries([rest_pcd, all_vis_pcd, arm_mesh, completed_branch_mesh] + arrow_lst)

