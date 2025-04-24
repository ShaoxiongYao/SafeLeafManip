import rtde_control
import rtde_receive
import time
import numpy as np
from ..pts_utils import rotation_vector_to_rotation_matrix

from .comModbusRtu import communication, getCommand

class UR5Real:
    """
    A wrapper class round rtde_control and rtde_receive for UR5 robot.
    
    This class provides utility functions to control the UR5 robot and
    communicate with the gripper.
    It includes functions to get and set angles, set payload, set TCP pose,
    get end effector pose, solve inverse kinematics, and execute motion.
    It also provides functions to connect and disconnect from the gripper.
    
    Attributes:
        ur5_ip (str): The IP address of the UR5 robot.
        rtde_c (RTDEControlInterface): The RTDE control interface for the UR5 robot.
        rtde_r (RTDEReceiveInterface): The RTDE receive interface for the UR5 robot.
        gripper_client (communication): The communication client for the gripper.
    """
    def __init__(self, ur5_ip, control=True, receive=True, gripper_serial=None):
        self.ur5_ip = ur5_ip

        if control:
            self.rtde_c = rtde_control.RTDEControlInterface(ur5_ip)
            time.sleep(0.1) # wait for the connection to be established

        if receive:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(ur5_ip)
            time.sleep(0.1) # wait for the connection to be established
        
        if gripper_serial is not None:
            self.gripper_client = communication(retry=False)
            self.gripper_client.connectToDevice(gripper_serial)
    
    def __del__(self):
        """
        Disconnect from the UR5 robot and gripper.
        """
        if hasattr(self, 'rtde_c'):
            self.rtde_c.disconnect()
        if hasattr(self, 'rtde_r'):
            self.rtde_r.disconnect()
        if hasattr(self, 'gripper_client'):
            self.gripper_client.disconnectFromDevice()

    def get_angles(self):
        """
        Get current joint angles of the UR5 robot.
        
        Returns:
            list: A list of joint angles in radians.
        """
        return self.rtde_r.getActualQ()
    
    def set_default_gripper_tcp(self):
        """
        Set the default TCP pose for the gripper.
        This function uses measured distance between end-effector 
        center to the gripper center, which is 0.1575m.
        """
        self.set_tcp_pose([0.0, 0.0, 0.1575, 0.0, 0.0, 0.0])

    def set_payload(self, mass, cog):
        """
        Set the payload for the UR5 robot.

        Args:
            mass (float): The mass of the payload in kg.
            cog (list): The center of gravity of the payload in [x, y, z]
                coordinates relative to the robot base frame.
        """
        self.rtde_c.setPayload(mass, cog)
    
    def set_default_gripper_payload(self):
        """
        Set the default payload for the gripper.
        
        This function uses the default payload of 0.9kg and a center of
        gravity at [0, 0, 0.05]m relative to the robot base frame.
        """
        self.set_payload(0.9, [0, 0, 0.05])

    def set_tcp_pose(self, pose):
        """
        Set the TCP pose for the UR5 robot.
        
        Args:
            pose (list): A list of 6 elements representing the TCP pose in
                [x, y, z, rx, ry, rz] format.
        """
        self.rtde_c.setTcp(pose)
    
    def get_ee_pose(self):
        """
        Get current end effector pose of the UR5 robot.
        
        Returns:
            list: A list of 6 elements representing the end effector pose
                in [x, y, z, rx, ry, rz] format.
        """
        return self.rtde_r.getActualTCPPose()
    
    def get_ee_trans_mat(self):
        """
        Get current end effector transformation matrix of the UR5 robot.
        
        The homogeneous matrix is of the form:
        | R11 R12 R13 X |
        | R21 R22 R23 Y |
        | R31 R32 R33 Z |
        | 0   0   0   1 |
        where R is the rotation matrix and X, Y, Z are the translation
        components.
        
        Returns:
            np.ndarray: A 4x4 transformation matrix representing the end
                effector pose.
        
        """
        tcp_pose = self.get_ee_pose()
        T = np.eye(4)
        T[:3, :3] = rotation_vector_to_rotation_matrix(tcp_pose[3:])
        T[:3, 3] = tcp_pose[:3]
        return T

    def gripper_execute(self, command, value=0):
        """
        Send a command to the gripper.
        Args:
            command (str): The command to send to the gripper.
            value (int): The value associated with the command.
        """
        message = getCommand(command, value=value)
        self.gripper_client.sendCommand(message)
    
    def solve_ik(self, pose, qnear=[], qlimits=None, max_position_error = 1e-10, max_orientation_error = 1e-10):
        """
        Use onboard inverse kinematics to solve for the joint angles
        that achieve the desired end effector pose.
        
        Args:
            pose (list): A list of 6 elements representing the desired end
                effector pose in [x, y, z, rx, ry, rz] format.
            qnear (list): A list of 6 elements representing the initial guess
                for the joint angles in [q1, q2, q3, q4, q5, q6] format.
            qlimits (list): A list of tuples representing the joint limits
                for each joint in the format [(q1_min, q1_max), (q2_min, q2_max), ...].
            max_position_error (float): The maximum position error allowed
                for the inverse kinematics solution.
            max_orientation_error (float): The maximum orientation error
                allowed for the inverse kinematics solution.

        Returns:
            list: A list of joint angles in radians that achieve the desired
                end effector pose, or an empty list if no solution is found.
        """
        # NOTE: angles unit is radian
        safe = self.rtde_c.isPoseWithinSafetyLimits(pose)
        if not safe:
            print ('INFO: pose is not within safety limits')
            return []
        else:
            has_ik_solution = self.rtde_c.getInverseKinematicsHasSolution(pose, qnear, max_position_error = max_position_error,
                                                                          max_orientation_error = max_orientation_error)
            if not has_ik_solution:
                print('INFO: no ik solution with max_position_error:', max_position_error, 'max_orientation_error:', max_orientation_error)
                return []
            else:
                angles = self.rtde_c.getInverseKinematics(pose, qnear, max_position_error = max_position_error, 
                                                          max_orientation_error = max_orientation_error)
                if qlimits is not None:
                    for q, (qmin, qmax) in zip(angles, qlimits):
                        # print ('q:', q, 'qmin:', qmin, 'qmax:', qmax)
                        if q < qmin or q > qmax:
                            print('INFO: joint limits violated')
                            return []
                forward_pose = self.rtde_c.getForwardKinematics(angles, self.rtde_c.getTCPOffset())
                print('INFO: forward_pose', forward_pose)
                print('expected pose:', pose)
                return angles
    
    def solve_ik_sequence(self, pose_lst, qnear=[], verbose=False):
        """
        Solve inverse kinematics for a sequence of poses.
        
        Args:
            pose_lst (list): A list of poses, where each pose is a list of
                6 elements representing the desired end effector pose in
                [x, y, z, rx, ry, rz] format.
            qnear (list): A list of 6 elements representing the initial guess
                for the joint angles in [q1, q2, q3, q4, q5, q6] format.
            verbose (bool): If True, print the pose and angles for each
                iteration.
        
        Returns:
            list: A list of joint angles in radians for each pose in the
                sequence, or an empty list if no solution is found for any
                pose.
        """
        angles_lst = []
        for pose in pose_lst:
            print('solve ik pose:', pose)
            angles = self.solve_ik(pose, qnear)
            if len(angles) == 0:
                return []
            else:
                angles_lst.append(angles)
                qnear = angles
        return angles_lst        
    
    def execute_motion(self, ee_pose=None, angles=None, motion_mode=None):
        """
        Have the real robot execute a motion in real-world space.
        
        !!!Be careful when executing motion on the real robot!!!
        !!!Make sure have the robot in a safe state before executing motion!!!
        !!!Make sure the safety stop is reachable place!!!
        
        Args:
            ee_pose (list): A list of 6 elements representing the desired end
                effector pose in [x, y, z, rx, ry, rz] format.
            angles (list): A list of joint angles in radians.
            motion_mode (str): The motion mode to use. Can be one of
                'CartesianLinear', 'JointLinear', 'CartesianFK', or 'JointIK'.
                - 'CartesianLinear': Move in Cartesian space.
                - 'JointLinear': Move in joint space.
                - 'CartesianFK': Move in Cartesian space using forward
                    kinematics.
                - 'JointIK': Move in joint space using inverse kinematics.
        """
        if isinstance(ee_pose, np.ndarray):
            ee_pose = ee_pose.tolist()
        if isinstance(angles, np.ndarray):
            angles = angles.tolist()

        if motion_mode == 'CartesianLinear':
            self.rtde_c.moveL(ee_pose, 0.25, 0.5, False)
        elif motion_mode == 'JointLinear':
            self.rtde_c.moveJ(angles, 1.05, 1.0, False)
        elif motion_mode == 'CartesianFK':
            self.rtde_c.moveL_FK(angles, 0.25, 0.5, False)
        elif motion_mode == 'JointIK':
            self.rtde_c.moveJ_IK(ee_pose, 1.05, 1.0, False)
