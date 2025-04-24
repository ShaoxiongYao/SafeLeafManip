import klampt
import numpy as np
import open3d as o3d
from klampt import WorldModel
from copy import deepcopy
from klampt.io import open3d_convert, numpy_convert
from ..vis_utils import gen_box
from pathlib import Path

class UR5Sim:
    """
    A wrapper class for the UR5 robot simulation using Klampt.
    """

    def __init__(self, robot_file, num_pcd_pts=0):
        self.world = WorldModel()
        self.world.readFile(robot_file)
        self.robot = self.world.robot(0)

        # Default: end effector is the 6th link
        self.ee_link = 7

        if num_pcd_pts > 0:
            self.save_arm_pcd(num_pts=num_pcd_pts)

    def set_angles(self, angles):
        """
        Set the angles of the robot's joints.
        
        Args:
            angles (list): A list of joint angles to set, expected to be of length 6.
        """
        q = self.robot.getConfig()
        q[1:1+len(angles)] = angles
        self.robot.setConfig(q)
        # self.robot.setConfig([0.0] + angles + [0.0])
    
    def get_angles(self):
        """
        Get the angles of the robot's joints.
        
        Returns:
            list: A list of joint angles, expected to be of length 6.
        """
        return self.robot.getConfig()[1:-1]
    
    def close_gripper(self):
        """
        Close the robot's gripper.
        This function sets the gripper to a closed position.
        
        TODO: values are hardcoded using klampt_control
        """
        q = self.robot.getConfig()
        assert len(q) == 17
        # NOTE: values are hardcoded using klampt_control
        q[-10:] = [0.0, 0.0, 0.723, 0.0, 0.723, -0.723, -0.723, 0.723, -0.723, 0.0]
        self.robot.setConfig(q)
    
    def open_gripper(self):
        """
        Open the robot's gripper.
        This function sets the gripper to an open position.
        
        TODO: values are hardcoded using klampt_control
        """
        q = self.robot.getConfig()
        assert len(q) == 17
        q[-10:] = [0.0] * 10
        self.robot.setConfig(q)
    
    def set_gripper_config(self, config_id):
        """
        Set the gripper configuration based on the given config_id.
        
        Four configurations [0-3] are available indicating increasing gripper closure.
        0: open, 1: 1/4 closed, 2: 1/2 closed, 3: fully closed.
        
        TODO: remove hardcoded values
        
        Args:
            config_id (int): The configuration ID for the gripper.
        """
        q = self.robot.getConfig()
        if config_id == 0:
            q[-10:] = [0.0] * 10
        elif config_id == 1:
            q[-10:] = [0.0, 0.0, 0.182162109375, 0.0, 0.182162109375, -0.182162109375, -0.182162109375, 0.182162109375, -0.182162109375, 0.0]
        elif config_id == 2:
            q[-10:] = [0.0, 0.0, 0.4441083984375, 0.0, 0.4441083984375, -0.4441083984375, -0.4441083984375, 0.4441083984375, -0.4441083984375, 0.0]
        elif config_id == 3:
            q[-10:] = [0.0, 0.0, 0.723, 0.0, 0.723, -0.723, -0.723, 0.723, -0.723, 0.0]
        self.robot.setConfig(q)
    
    def save_arm_pcd(self, num_pts=5000):
        """
        Save the point cloud of the robot arm in self.local_pcd_lst.
        
        Args:
            num_pts (int): The number of points to sample from each link's geometry.
        """
        self.local_pcd_lst = []

        for link_idx in range(self.robot.numLinks()):
            link_i = self. robot.link(link_idx)
            link_geom = link_i.geometry()

            mesh = open3d_convert.to_open3d(link_geom.getTriangleMesh())
            # compute normal from mesh is necessary
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()

            pcd = mesh.sample_points_uniformly(number_of_points=num_pts, 
                                           use_triangle_normal=True) 
            self.local_pcd_lst.append(pcd)

        return self.local_pcd_lst
    
    def save_arm_info(self, out_dir:Path, suffix:str, angles=True, arm_pcd=False, box_pcd=True):
        """
        Save the robot arm information including angles, arm point cloud, and box point cloud
        to the specified output directory with the given suffix.
        
        Args:
            out_dir (Path): The output directory to save the information.
            suffix (str): The suffix to append to the saved files.
            angles (bool): Whether to save the angles of the robot's joints.
            arm_pcd (bool): Whether to save the arm point cloud.
            box_pcd (bool): Whether to save the box point cloud.
        """
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        if angles:
            np.save(out_dir / f'angles_{suffix}.npy', self.get_angles())
        
        if arm_pcd:
            arm_pcd = self.get_arm_pcd()
            o3d.io.write_point_cloud(str(out_dir / f'arm_pcd_{suffix}.ply'), arm_pcd)

        if box_pcd:
            box_pcd = self.get_box_pcd()
            o3d.io.write_point_cloud(str(out_dir / f'box_pcd_{suffix}.ply'), box_pcd)

    def get_trans_lst(self):
        """Get transformation matrix for each link."""
        link_trans_lst = []
        for link_idx in range(self.robot.numLinks()):

            link_trans = self.robot.link(link_idx).getTransform()
            np_link_trans = numpy_convert.to_numpy(link_trans)
            link_trans_lst.append(np_link_trans)

        return link_trans_lst
    
    def get_ee_trans(self):
        """Get transformation matrix for end effector."""
        return self.get_trans_lst()[self.ee_link]
    
    def get_arm_pcd(self, link_idx_lst=None):
        """Get current arm point cloud in world frame."""
        if link_idx_lst is None:
            link_idx_lst = range(self.robot.numLinks())
        
        world_pcd_lst = []
        for link_idx in link_idx_lst:
            local_pcd = deepcopy(self.local_pcd_lst[link_idx])
            H = self.get_trans_lst()[link_idx]
            world_pcd_lst.append(local_pcd.transform(H))

        return sum(world_pcd_lst, o3d.geometry.PointCloud())
    
    def get_mesh_lst(self, link_idx_lst=None, compute_normals=True):
        """Get current arm mesh in world frame."""
        if link_idx_lst is None:
            link_idx_lst = range(self.robot.numLinks())
        
        world_mesh_lst = []
        for link_idx in link_idx_lst:
            link_i = self.robot.link(link_idx)
            link_geom = link_i.geometry()

            H = self.get_trans_lst()[link_idx]
            mesh = open3d_convert.to_open3d(link_geom.getTriangleMesh())
            if compute_normals:
                mesh.compute_vertex_normals()

            world_mesh_lst.append(mesh.transform(H))
        return world_mesh_lst

    def get_gripper_open2close_pcd(self):
        """
        Get a set of point clouds for gripper from open to close.
        
        Returns:
            pcd (o3d.geometry.PointCloud): The summed point cloud of arm + gripper.
        """
        arm_pcd = self.get_arm_pcd(link_idx_lst=[0, 1, 2, 3, 4, 5, 6])
        gripper_pcd = o3d.geometry.PointCloud()
        curr_config = self.robot.getConfig()
        for config_id in range(4):
            self.set_gripper_config(config_id)
            gripper_pcd += self.get_arm_pcd(link_idx_lst=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.robot.setConfig(curr_config)
        return arm_pcd + gripper_pcd

    def get_box_pcd(self, color=[0.0, 0.0, 1.0], offset=[0.1575, 0.0, 0.0]):
        """
        Get a box point cloud in the end effector frame represented the points to be grasped.
        
        Args:
            color (list): The color of the box point cloud.
            offset (list): The offset to be applied to the box point cloud.
        Returns:
            box_pcd (o3d.geometry.PointCloud): The box point cloud in the end effector frame.
        """
        box_pts = gen_box(0.025, 25)
        box_pcd = o3d.geometry.PointCloud()
        box_pcd.points = o3d.utility.Vector3dVector(box_pts)
    
        box_pcd.paint_uniform_color(color)

        box_pcd.translate(offset)
        box_pcd.transform(self.get_ee_trans())
        return box_pcd
