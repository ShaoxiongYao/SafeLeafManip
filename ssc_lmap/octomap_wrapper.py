
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