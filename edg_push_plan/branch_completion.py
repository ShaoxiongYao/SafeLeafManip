import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import sys
import io
from copy import deepcopy

import matplotlib.pyplot as plt

from .vis_utils import create_cylinder_between_two_points, create_ball, create_motion_lines, create_arrow_lst, display_inlier_outlier
from .pts_utils import distance_point_to_line, average_distance_points_to_line

class OutputCapturer(io.StringIO):
    """A context manager to capture stdout output"""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.output = self.getvalue()
        sys.stdout = self._stdout

def parse_energy(output, target_iter = 0):
    """A utility function to parse the energy value from the output of Open3D
    ARAP deformation, which is printed to stdout.
    
    Args:
        output (str): The output string from Open3D.
        target_iter (int): The iteration number to look for.
    Returns:
        float: The energy value at the specified iteration.
    
    """
    for line in output.splitlines():
        if "[DeformAsRigidAsPossible] iter" in line:
            parts = line.split(' ')
            iter_value = None
            energy_value = None
            for part in parts:
                if part.startswith('iter='):
                    iter_value = int(part.split('=')[1].rstrip(','))
                if part.startswith('energy='):
                    energy_value = float(part.split('=')[1])
            if iter_value == target_iter:
                return energy_value
    return None

class BranchCompletion:
    """
    Complete branch shape from observed partial point cloud using
    cylinder triangle mesh and as-rigid-as-possible deformation
    """
    def __init__(self, cylinder_resolution, min_cross_num=5, radius_rate=0.9, cylinder_movement_rate=2.0) -> None:
        """
        Args:
            cylinder_resolution: float, the resolution of the cylinder mesh
            min_cross_num: int, minimum number of cross points to consider
            radius_rate: float, rate to adjust the average distance for cylinder radius
            cylinder_movement_rate: float, rate to adjust the cylinder movement towards camera center
        
        """
        self.cylinder_resolution = cylinder_resolution
        self.min_cross_num = min_cross_num
        self.radius_rate = radius_rate
        self.cylinder_movement_rate = cylinder_movement_rate

    def predict_initial_cylinder(self, branch_pcd, cam_center, verbose=False, vis_check=False, return_axis=False):
        """
        Predict initial cylinder shape from the branch point cloud
        
        Args:
            branch_pcd: open3d.geometry.PointCloud, branch point cloud
            cam_center: camera center position
            verbose: print debug information
            vis_check: visualize the predicted cylinder
            return_axis: return the axis of the cylinder
        """

        points = np.asarray(branch_pcd.points)

        # Run PCA on branch points, assume the largest variance direction as the axis
        pca = PCA(n_components=1)
        pca.fit(points)
        axis = pca.components_[0]

        # Flip the axis if points downward z-axis
        if axis[2] < 0:
            axis = -axis

        project_distance = np.dot(points, axis)
        start_idx, end_idx = np.argmin(project_distance), np.argmax(project_distance)

        start_pt = points[start_idx, :]
        end_pt = points[end_idx, :]
        if verbose:
            print('end-start axis:', end_pt - start_pt)

        radius_lst = []
        for mid_pt in np.linspace(start_pt, end_pt, num=20):
            pca_val = np.dot(mid_pt, axis)
            print('pca_val:', pca_val)

            # Cross points
            cross_idx = np.where(np.abs(project_distance - pca_val) <= 0.01)[0]
            if len(cross_idx) <= self.min_cross_num:
                continue
            cross_pts = points[cross_idx, :]

            # cross_pcd = o3d.geometry.PointCloud()
            # cross_pcd.points = o3d.utility.Vector3dVector(cross_pts)
            # cross_pcd.paint_uniform_color([1, 0, 0])
            # cross_pcd.translate([0.0, 0.0, -0.0001])
            # o3d.visualization.draw_geometries([cross_pcd, branch_pcd])

            center_pts = np.mean(cross_pts, axis=0)
            radius = np.mean(np.linalg.norm(cross_pts - center_pts, axis=1))

            radius_lst.append(radius)
        print('radius_lst:', radius_lst)
        
        avg_dist = np.mean(radius_lst) * self.radius_rate


        # avg_dist = average_distance_points_to_line(points, start_pt, end_pt)
        # if verbose:
        #     print('avg_dist:', avg_dist)
   
        resolution = int(np.ceil(2*np.pi*avg_dist / self.cylinder_resolution))
        split = int(np.ceil(np.linalg.norm(end_pt - start_pt) / self.cylinder_resolution))
        if verbose:
            print('resolution:', resolution)
            print('split:', split)

        # cylinder1 = create_cylinder_between_two_points(start_pt, (start_pt+end_pt)/2, avg_dist, 
        #                                                resolution=resolution, split=split)
        # cylinder2 = create_cylinder_between_two_points((start_pt+end_pt)/2, end_pt, avg_dist,
        #                                                resolution=resolution, split=split)
        # cylinder = cylinder1 + cylinder2
        cylinder = create_cylinder_between_two_points(start_pt, end_pt, avg_dist,
                                                      resolution=resolution, split=split)
        cylinder.compute_vertex_normals()
        cylinder_pcd = o3d.geometry.PointCloud()
        cylinder_pcd.points = o3d.utility.Vector3dVector(np.asarray(cylinder.vertices))

        distance, shortest_vec = distance_point_to_line(cam_center, start_pt, end_pt, 
                                                        ret_shortest_vec=True)
        cylinder_pcd.translate(-self.cylinder_movement_rate*avg_dist*shortest_vec)
        cylinder.translate(-self.cylinder_movement_rate*avg_dist*shortest_vec)

        if vis_check:
            camera_ball = create_ball(radius=0.01, color=[0, 1, 0], center=cam_center)
            o3d.visualization.draw_geometries([camera_ball, cylinder_pcd, branch_pcd])
        
        if not return_axis:
            return cylinder, end_pt - start_pt
        else:
            return cylinder, end_pt, start_pt
    
    def get_cylinder_front_back_pts(self, cylinder_mesh, cylinder_axis, cam_center, vis_check=False):
        all_pts = np.asarray(cylinder_mesh.vertices)

        cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

        center2cam_vec = cam_center - cylinder_mesh.get_center()
        center2cam_vec = center2cam_vec / np.linalg.norm(center2cam_vec)
        side_vec = np.cross(cylinder_axis, center2cam_vec)
        side_vec = side_vec / np.linalg.norm(side_vec)

        normal_vec = np.cross(side_vec, cylinder_axis)
        normal_vec = normal_vec / np.linalg.norm(normal_vec)

        p1_ary = np.array([cylinder_mesh.get_center(), cylinder_mesh.get_center(), cylinder_mesh.get_center()])
        p2_ary = np.array([cylinder_mesh.get_center() + cylinder_axis, 
                           cylinder_mesh.get_center() + side_vec, cylinder_mesh.get_center() + normal_vec])

        arrow_lst = create_arrow_lst(p1_ary, p2_ary, scale=0.3, color=[0, 0.5, 0.8])

        vertex2cam_dot = np.dot(all_pts-cylinder_mesh.get_center(), normal_vec)

        if vis_check:
            plt.hist(vertex2cam_dot.flatten(), bins=50)
            plt.show()

        front_pts = all_pts[vertex2cam_dot.flatten() > 0, :]
        back_pts = all_pts[vertex2cam_dot.flatten() <= 0, :]
        
        if vis_check:
            front_pcd = o3d.geometry.PointCloud()
            front_pcd.points = o3d.utility.Vector3dVector(front_pts)
            front_pcd.paint_uniform_color([0, 0, 1])
            back_pcd = o3d.geometry.PointCloud()
            back_pcd.points = o3d.utility.Vector3dVector(back_pts)
            back_pcd.paint_uniform_color([1, 0, 0])
            camera_ball = create_ball(radius=0.01, color=[0, 1, 0], center=cam_center)
            o3d.visualization.draw_geometries([front_pcd, back_pcd, camera_ball] + arrow_lst)

        front_vertices_mask = vertex2cam_dot.flatten() > 0
        front_vertices_idx = np.where(front_vertices_mask)[0]
        back_vertices_idx = np.where(~front_vertices_mask)[0]
        return front_vertices_idx, back_vertices_idx

    def complete(self, branch_pcd, cam_center, voxel_size=0.003, match_dist_thresh=0.03, 
                 max_iteration=100, arap_max_iter=50, reg_front_pts=True, smooth_branch=False, 
                 verbose=False, vis_check=False):
        """
        Complete branch shape from observed partial point cloud using deformable cylinder
        """
        branch_pcd_copy = deepcopy(branch_pcd)
        cl, ind = branch_pcd_copy.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
        display_inlier_outlier(branch_pcd_copy, ind)
        branch_pcd_copy = branch_pcd_copy.select_by_index(ind)

        ds_branch_pcd = branch_pcd_copy.voxel_down_sample(voxel_size=voxel_size)
        ds_branch_pts = np.asarray(ds_branch_pcd.points)

        deform_cylinder_mesh, axis = self.predict_initial_cylinder(ds_branch_pcd, cam_center, 
                                                                   verbose=verbose, vis_check=vis_check)

        front_vertices_idx, back_vertices_idx = self.get_cylinder_front_back_pts(deform_cylinder_mesh, axis, cam_center, 
                                                                                 vis_check=vis_check)
        if reg_front_pts:
            front_vertices_idx = front_vertices_idx
        else:
            front_vertices_idx = np.arange(len(deform_cylinder_mesh.vertices))

        for _ in range(max_iteration):
            # curr_cy_pts = np.asarray(deform_cylinder_mesh.vertices)
            all_cy_pts = np.asarray(deform_cylinder_mesh.vertices)

            curr_cy_pts = all_cy_pts[front_vertices_idx, :]

            curr_knn = NearestNeighbors(n_neighbors=1).fit(curr_cy_pts)
            eud, idx = curr_knn.kneighbors(ds_branch_pts)

            idx = idx[eud.flatten() < match_dist_thresh]
            match_cylinder_idx = front_vertices_idx[idx.flatten()]
            match_branch_pts = ds_branch_pts[eud.flatten() < match_dist_thresh, :]

            # match_cylinder_idx, unique_idx = np.unique(match_cylinder_idx, return_index=True)
            # match_branch_pts = match_branch_pts[unique_idx, :]

            cy_pts = curr_cy_pts[idx.flatten(), :]

            if vis_check:
                pcd1, pcd2, lines = create_motion_lines(match_branch_pts, cy_pts, return_pcd=True)
                o3d.visualization.draw_geometries([pcd1, pcd2, deform_cylinder_mesh, lines])
            
            constraint_ids = o3d.utility.IntVector(match_cylinder_idx.flatten().tolist())
            constraint_pos = o3d.utility.Vector3dVector(match_branch_pts)

            with OutputCapturer() as capturer:
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                    mesh_prime = deform_cylinder_mesh.deform_as_rigid_as_possible(constraint_ids,
                                                                                  constraint_pos,
                                                                                  max_iter=arap_max_iter)

            energy = parse_energy(capturer.output, target_iter=0)
            print('energy:', energy)

            mesh_prime.compute_vertex_normals()
            # o3d.visualization.draw_geometries([mesh_prime, branch_pcd])

            deform_cylinder_mesh = mesh_prime

            if energy < 1e-10:
                break
            
            # if smooth_branch:
            #     new_cylinder_mesh = deepcopy(deform_cylinder_mesh)
            #     new_cylinder_mesh = new_cylinder_mesh.filter_smooth_taubin(number_of_iterations=3)
            #     new_cylinder_mesh.compute_vertex_normals()
            #     new_cylinder_mesh.paint_uniform_color([0, 1, 0])
            #     if vis_check:
            #         o3d.visualization.draw_geometries([new_cylinder_mesh, deform_cylinder_mesh, branch_pcd])
            #     deform_cylinder_mesh = new_cylinder_mesh
            
        if smooth_branch:
            deform_cylinder_mesh = deform_cylinder_mesh.filter_smooth_taubin(number_of_iterations=3)
        
        return deform_cylinder_mesh
