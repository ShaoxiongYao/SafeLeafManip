import time
import multiprocessing as mp

import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d
import cv2
import json
import pickle

camera_model_dict = {
    0: rs.distortion.none,
    4: rs.distortion.brown_conrady,
}

def load_intrinsics_from_json(json_fn):
    """
    Load the camera intrinsics from a JSON file.

    Args:
        json_fn (str): The path to the JSON file.
        
    Returns:
        tuple: A tuple containing the color and depth camera intrinsics.
    """
    with open(json_fn, 'r') as f:
        intrinsics_dict = json.load(f)
        
        depth_intrinsic = rs.pyrealsense2.intrinsics()
        depth_intrinsic.width = intrinsics_dict['depth'][0]
        depth_intrinsic.height = intrinsics_dict['depth'][1]
        depth_intrinsic.ppx = intrinsics_dict['depth'][2]
        depth_intrinsic.ppy = intrinsics_dict['depth'][3]
        depth_intrinsic.fx = intrinsics_dict['depth'][4]
        depth_intrinsic.fy = intrinsics_dict['depth'][5]
        depth_intrinsic.model = camera_model_dict[intrinsics_dict['depth'][6]]
        depth_intrinsic.coeffs[0] = intrinsics_dict['depth'][7]
        depth_intrinsic.coeffs[1] = intrinsics_dict['depth'][8]
        depth_intrinsic.coeffs[2] = intrinsics_dict['depth'][9]
        depth_intrinsic.coeffs[3] = intrinsics_dict['depth'][10]
        depth_intrinsic.coeffs[4] = intrinsics_dict['depth'][11]

        color_intrinsic = rs.pyrealsense2.intrinsics()
        color_intrinsic.width = intrinsics_dict['color'][0]
        color_intrinsic.height = intrinsics_dict['color'][1]
        color_intrinsic.ppx = intrinsics_dict['color'][2]
        color_intrinsic.ppy = intrinsics_dict['color'][3]
        color_intrinsic.fx = intrinsics_dict['color'][4]
        color_intrinsic.fy = intrinsics_dict['color'][5]
        color_intrinsic.model = camera_model_dict[intrinsics_dict['color'][6]]
        color_intrinsic.coeffs[0] = intrinsics_dict['color'][7]
        color_intrinsic.coeffs[1] = intrinsics_dict['color'][8]
        color_intrinsic.coeffs[2] = intrinsics_dict['color'][9]
        color_intrinsic.coeffs[3] = intrinsics_dict['color'][10]
        color_intrinsic.coeffs[4] = intrinsics_dict['color'][11]

    return color_intrinsic, depth_intrinsic

def save_intrinsics_from_json(json_fn, color_intr, depth_intr):
    """
    Save the camera intrinsics to a JSON file.
    Args:
        json_fn (str): The path to the JSON file.
        color_intr (rs.pyrealsense2.intrinsics): The color camera intrinsics.
        depth_intr (rs.pyrealsense2.intrinsics): The depth camera intrinsics.
    """
    intrinsics_dict = {
        'color': [color_intr.width, color_intr.height, color_intr.ppx, color_intr.ppy, 
                  color_intr.fx, color_intr.fy, int(color_intr.model), 
                  color_intr.coeffs[0], color_intr.coeffs[1], color_intr.coeffs[2], 
                  color_intr.coeffs[3], color_intr.coeffs[4]],
        'depth': [depth_intr.width, depth_intr.height, depth_intr.ppx, depth_intr.ppy, 
                  depth_intr.fx, depth_intr.fy, int(depth_intr.model), 
                  depth_intr.coeffs[0], depth_intr.coeffs[1], depth_intr.coeffs[2], 
                  depth_intr.coeffs[3], depth_intr.coeffs[4]]
    }
    with open(json_fn, 'w') as f:
        json.dump(intrinsics_dict, f)
