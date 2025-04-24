import open3d as o3d
import numpy as np
import copy
from typing import List
from .pts_utils import pca_points, remove_close_points, sample_points_on_hull
from .vis_utils import plot_hull_pts, display_inlier_outlier

from dataclasses import dataclass

@dataclass
class GraspPlannerConfig:
    """
    Configurations for the grasp planner.
    """
    obstacle_distance_threshold: float = 0.06
    """ Minimum distance to the obstacle point cloud."""
    object_boundary_threshold: float = 0.01
    """ Distance to the boundary of the leaf object."""
    approach_distance: float = 0.01
    """ Distance from the grasp point to the approach point."""
    num_cvx_samples: int = 100
    """ Number of samples to take on the convex hull of the object."""

@dataclass
class PullActionConfig:
    """
    Configurations for the pull action.
    """
    move_min_dist: float
    """ Minimum distance for the pull action."""
    move_max_dist: float
    """ Maximum distance for the pull action."""
    move_steps: int
    """ Number of steps for the pull action."""

class GraspPlanner:
    """
    Class to plan grasps for a leaf point cloud.
    
    This grasp planner uses heuristic to detect edge points of a leaf point cloud 
    and have sampled 6DoF grasp poses pointing towards the centroid of the leaf.
    
    Attributes:
        obstacle_pcd (o3d.geometry.PointCloud): Point cloud of the environment, the planned grasp will avoid this point cloud.
        obstacle_distance_threshold (float): Minimum distance to the obstacle point cloud.
        object_boundary_threshold (float): Distance to the boundary of the leaf object
        approach_distance (float): Distance from the grasp point to the approach point.
    """
    def __init__(self, obstacle_pcd, config: GraspPlannerConfig):
        # env_pcd is the point cloud of the environment
        self.obstacle_pcd = obstacle_pcd
        
        self.obstacle_distance_threshold = config.obstacle_distance_threshold
        self.object_boundary_threshold = config.object_boundary_threshold
        self.approach_distance = config.approach_distance
        self.num_cvx_samples = config.num_cvx_samples
        # self.obstacle_distance_threshold = obstacle_distance_threshold
        # self.object_boundary_threshold = object_boundary_threshold
        # self.approach_distance = approach_distance
    
    def sample_plane_points(self, obj_pcd, num_cvx_samples=100, vis_cvx_hull=False):
        """
        Sample edge points on approximated 2D plane of the input object point cloud.
        
        This function follows the following steps:
        1. Run PCA on the point cloud to get the centroid and principal axes.
        2. Obtain the PCA components of the point cloud by projecting the points 
        onto the PCA axes as aligned_points (Nx3).
        3. Take the first two components of the aligned_points as 2D boundary points, 
        and fit a 2D convex hull to the points and sample points on the convex hull.
        4. Move the sampled points inward the convex hull by a distance of
        object_boundary_threshold.
        5. Remove grasp points that are close to the obstacle point cloud, defined by 
        the distance threshold obstacle_distance_threshold.
        6. Use the PCA normal of the point cloud as the normals of the sampled points.
        
        Args:
            obj_pcd (o3d.geometry.PointCloud): Point cloud of the object to grasp.
            num_cvx_samples (int): Number of samples to take on the convex hull of the object.
            vis_cvx_hull (bool): Whether to visualize the convex hull and the sampled points.
        
        Returns:
            dict: A dictionary containing the following keys:
                - 'clean_sampled_pcd': The cleaned sampled point cloud after removing close points.
                - 'centroid': The centroid of the object point cloud.
                - 'plane_normal': The normal of the plane fitted to the object point cloud.
        """

        points = np.asarray(obj_pcd.points)
        centroid, eigen_values, eigen_vectors = pca_points(points)

        # Get the normal of the plane
        plane_normal = eigen_vectors[:, 2]
        if plane_normal[2] > 0:
            plane_normal *= -1

        if vis_cvx_hull:
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            world_frame.translate(centroid)

            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coord_frame.rotate(eigen_vectors.T)
            coord_frame.translate(centroid)
            o3d.visualization.draw_geometries([obj_pcd, coord_frame, world_frame], point_show_normal=True)

        # Rotate the point cloud to align with the principal axes
        aligned_points = np.dot(points - centroid, eigen_vectors)

        # Sample points on the convex hull of the object 
        sample_dict = sample_points_on_hull(aligned_points[:, :2], num_cvx_samples)
        hull, sampled_points = sample_dict['hull'], sample_dict['sampled_points']

        # Move points inward the convex hull
        move_direction = np.mean(sampled_points, axis=0) - sampled_points
        move_direction /= np.linalg.norm(move_direction, axis=1)[:, None]
        sampled_points += self.object_boundary_threshold * move_direction

        if vis_cvx_hull:
            plot_hull_pts(hull, aligned_points, sampled_points)

        # Augment the sampled points with z values
        sampled_points = np.hstack([sampled_points, np.zeros((sampled_points.shape[0], 1))])

        # Transform the sampled points back to the original frame
        trans_sampled_points = np.dot(sampled_points, eigen_vectors.T) + centroid
        trans_sampled_pcd = o3d.geometry.PointCloud()
        trans_sampled_pcd.points = o3d.utility.Vector3dVector(trans_sampled_points)
        trans_sampled_pcd.paint_uniform_color([0, 1, 0])

        # Remove grasp points close to obstacle
        clean_sampled_pcd = remove_close_points(self.obstacle_pcd, trans_sampled_pcd, 
                                                self.obstacle_distance_threshold)
        
        # Use PCA as sampled point cloud normals
        normals = np.repeat(plane_normal.reshape(1, -1), len(clean_sampled_pcd.points), axis=0)
        clean_sampled_pcd.normals = o3d.utility.Vector3dVector(normals)
        
        return {
            'clean_sampled_pcd': clean_sampled_pcd,
            'centroid': centroid, 'plane_normal': plane_normal
        }

    def vis_approach_direction(self, clean_sampled_pcd, approach_points, env_pcd):
        """
        Utility function to visualize the approach direction of the grasp.
        
        Args:
            clean_sampled_pcd (o3d.geometry.PointCloud): Clean sampled point cloud.
            approach_points (np.ndarray): Approach points.
            env_pcd (o3d.geometry.PointCloud): Environment point cloud. 
        """
        clean_sampled_points = np.asarray(clean_sampled_pcd.points)

        # Visualize the approach direction
        approach_pcd = o3d.geometry.PointCloud()
        approach_pcd.points = o3d.utility.Vector3dVector(approach_points)
        approach_pcd.paint_uniform_color([1, 0, 0])

        num_pts = len(approach_points)

        # Draw line set between the sampled points and the approach points
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack([clean_sampled_points, approach_points]))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i+num_pts] for i in range(num_pts)]))
        o3d.visualization.draw_geometries([clean_sampled_pcd, approach_pcd, env_pcd, line_set], point_show_normal=True)

    def plan_grasp_leaf(self, leaf_pcd, vis_cvx_hull=False, 
                        vis_approach_pts=False, vis_grasp_frame=False, 
                        vis_remove_outliers=False) -> List[np.ndarray]:
        """
        Take the point cloud of the leaf and plan the grasp frames
        Args: 
            leaf_pcd (open3d.geometry.PointCloud): Point cloud of the leaf to grasp
            num_cvx_samples (int): Number of samples to take on the convex hull of the leaf.
            vis_cvx_hull (bool): Whether to visualize the convex hull.
            vis_approach_pts (bool): Whether to visualize the approach points.
            vis_grasp_frame (bool): Whether to visualize the grasp frames.
            vis_remove_outliers (bool): Whether to visualize the removal of outliers.

        Returns:
            grasp_frame_lst (list of np.ndarray): List of grasp frames in the form of 4x4 transformation matrices.
        """
        # Remove statistical outliers of the leaf point cloud
        leaf_pcd_copy = copy.deepcopy(leaf_pcd)
        cl, ind = leaf_pcd_copy.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
        if vis_remove_outliers:
            display_inlier_outlier(leaf_pcd_copy, ind)
        leaf_pcd_copy = leaf_pcd_copy.select_by_index(ind)

        # sample grasp points on fitted plane
        plane_info_dict = self.sample_plane_points(leaf_pcd_copy, self.num_cvx_samples, vis_cvx_hull=vis_cvx_hull)
        clean_sampled_pcd = plane_info_dict['clean_sampled_pcd']
        centroid, plane_normal = plane_info_dict['centroid'], plane_info_dict['plane_normal']

        # compute approach direction
        clean_sampled_points = np.asarray(clean_sampled_pcd.points)
        approach_direction = centroid - clean_sampled_points
        approach_direction /= np.linalg.norm(approach_direction, axis=1)[:, None]

        # compute approach points
        approach_points = clean_sampled_points + self.approach_distance * approach_direction
        if vis_approach_pts:
            self.vis_approach_direction(clean_sampled_pcd, approach_points, leaf_pcd)

        # compute grasp frames
        grasp_frame_lst = []
        for grasp_id in range(len(approach_points)):
            start_position = clean_sampled_points[grasp_id, :]
            end_position = approach_points[grasp_id, :]  

            # Translation matrix T
            T = np.eye(4)
            T[:3, 3] = start_position

            # Calculating vector X
            X = -plane_normal
            X = X / np.linalg.norm(X)

            # Calculating vector Z
            Z = end_position - start_position 
            Z = Z / np.linalg.norm(Z)

            # Calculating vector Y
            Y = np.cross(Z, X)
            Y = Y / np.linalg.norm(Y)

            T[:3, :3] = np.vstack([X, Y, Z]).T
            grasp_frame_lst.append(T)

            if vis_grasp_frame:
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                coord_frame.transform(T)
                o3d.visualization.draw_geometries([clean_sampled_pcd, leaf_pcd, coord_frame], point_show_normal=True)

        return grasp_frame_lst