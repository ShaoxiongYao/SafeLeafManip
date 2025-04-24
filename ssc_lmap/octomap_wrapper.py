
import time
import numpy as np
import open3d as o3d
import octomap

from .pts_utils import is_free_frontier_point

from dataclasses import dataclass

@dataclass
class FreeSpaceConfig:
    """
    Configurations to compute free space points around the fruit.
    Attributes:
        voxel_size (float): The voxel size for the octomap.
        free_space_bbx_scale (float): The scale for the bounding box of the free space.
        free_space_layer (int): The layer of the free space.
        fruit_max_size (float): The maximum size of the fruit.
    """
    voxel_size: float
    free_space_bbx_scale: float
    free_space_layer: int
    fruit_max_size: float

class OctomapWrapper:
    """
    A wrapper class for octomap to compute free space points and visible points.
    This class provides utility functions to compute free space points around the fruit
    and visible points in the octomap.
    """
    
    def __init__(self):
        pass
    
    def compute_free_space_points(self, raw_scene_pcd: o3d.geometry.PointCloud, 
                                  fruit_pts: np.ndarray, cam_center: np.ndarray, 
                                  voxel_size: float, fruit_max_size: float, 
                                  free_space_bbx_scale: float, free_space_layer: int, 
                                  verbose: bool=False) -> np.ndarray:
        """
        A utility function to compute the free space points around the fruit.
        
        Args:
            raw_scene_pcd (o3d.geometry.PointCloud): The raw scene point cloud.
            fruit_pts (np.ndarray): shape (N, 3), the fruit point cloud. 
            cam_center (np.ndarray): shape (3,), the camera center.
            voxel_size (float): The voxel size for the octomap.
            fruit_max_size (float): The maximum size of the fruit.
            free_space_bbx_scale (float): The scale for the bounding box of the free space.
            free_space_layer (int): The layer of the free space.

        Returns:
            free_space_pts (np.ndarray): shape (N_free, 3), the free space points around the fruit.
        """

        # voxel downsample the raw scene point cloud
        raw_scene_pcd = raw_scene_pcd.voxel_down_sample(voxel_size=voxel_size)
        
        time_free_space_start = time.time()
        
        # compute the center of the fruit
        fruit_center = np.mean(fruit_pts, axis=0)
        
        #create an OctoMap
        free_octomap_scene = octomap.OcTree(voxel_size)

        #compute free space are close to object
        coord_range = int(fruit_max_size / voxel_size * free_space_bbx_scale)

        #initialize unknown space
        for x in range(-coord_range, coord_range):
            for y in range(-coord_range, coord_range):
                for z in range(-coord_range, coord_range):
                    pts = fruit_center + np.array([x*voxel_size, y*voxel_size, z*voxel_size])
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
                    pts = fruit_center + np.array([x*voxel_size, y*voxel_size, z*voxel_size])
                    if np.linalg.norm(pts - fruit_center) > 2.0 * fruit_max_size * free_space_bbx_scale:
                        continue
                    if is_free_frontier_point(pts, free_octomap_scene, voxel_size = voxel_size, layer = free_space_layer):
                        free_space_lst.append(pts)

        free_space_pts = np.array(free_space_lst)
        
        time_free_space_end = time.time()
        
        if verbose:
            print('free space pcd created')
            print ('free space time:', time_free_space_end - time_free_space_start)
        
        return free_space_pts

    def compute_visible_points(self, merged_pcd: o3d.geometry.PointCloud,
                               fit_fruit_pcd: o3d.geometry.PointCloud, cam_center: np.ndarray,
                               voxel_size: float, verbose: bool=False) -> np.ndarray:
        """
        A utility function to compute the visible points in the octomap.
        
        This function first creates an octomap and insert merged_pcd into it.
        Then it uses ray tracing to check if the points in fit_fruit_pcd are visible from the camera center.
        The function returns the visible points and the octomap points.
        
        Args:
            merged_pcd (o3d.geometry.PointCloud): The merged point cloud.
            fit_fruit_pcd (o3d.geometry.PointCloud): The fitted fruit point cloud.
            cam_center (np.ndarray): shape (3,), the camera center.
            voxel_size (float): The voxel size for the octomap.
            verbose (bool): If True, print the time taken for ray tracing.
        
        Returns:
            vis_points (np.ndarray): shape (N_vis, 3), the visible points.
            octo_pts (np.ndarray): shape (N_octo, 3), the octomap points.
        """

        # copy a octomap scene to add fit points and arm pcd for ray tracing 
        octomap_scene_copy = octomap.OcTree(voxel_size)
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
        if verbose:
            print('ray tracing time:', time.time() - start_time)
        
        octo_pts = np.array(octo_pts)
    
        vis_points = []
        for k in vis_dict.keys():
            octo_k = octomap.OcTreeKey()
            octo_k[0] = k[0]
            octo_k[1] = k[1]
            octo_k[2] = k[2]
            vis_points.append(octomap_scene_copy.keyToCoord(octo_k))
        vis_points = np.array(vis_points)
        
        return vis_points, octo_pts