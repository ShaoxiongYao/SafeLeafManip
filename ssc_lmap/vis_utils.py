import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def image_black2white(image):
    """
    Converts black pixels in an image to white.
    
    Args:
        image (numpy.ndarray): shape (H, W, 3), the input RGB image.
    
    Returns:
        numpy.ndarray: The modified image with black pixels converted to white.    
    """
    black_mask = np.all(image == 0, axis=-1)
    image[black_mask] = [255, 255, 255]
    return image

def bool2color(bool_array, color_true=[0, 1, 0], color_false=[1, 0, 0]):
    colors = np.zeros((bool_array.shape[0], 3))
    colors[bool_array] = color_true
    colors[~bool_array] = color_false
    return colors

def image_plane2pcd(image, w, h, z=0):
    img_x, img_y = np.meshgrid(np.arange(w), np.arange(h))
    img_pts = np.vstack([img_x.flatten(), img_y.flatten()]).T
    img_pts = np.concatenate([img_pts, z*np.ones((img_pts.shape[0], 1))], axis=1)

    img_pcd = o3d.geometry.PointCloud()
    img_pcd.points = o3d.utility.Vector3dVector(img_pts)
    img_pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)

    return img_pcd

def plot_hull_pts(hull, hull_points, sampled_points):
    # Plot the convex hull
    for simplex in hull.simplices:
        plt.plot(hull_points[simplex, 0], hull_points[simplex, 1], 'k-')  # Plot edges with black lines
    plt.plot(sampled_points[:, 0], sampled_points[:, 1], 'r.')  # Plot sampled points as red dots
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def scalars_to_colors(input_array, colormap='viridis', min_val=None, max_val=None):
    """
    Maps a 1-dimensional numpy array (shape (n,)) of scalars to a 3xn numpy array of RGB colors.
    
    Parameters:
    - input_array: A 1-dimensional numpy array of scalars (shape (n,)).
    - colormap: A string specifying the Matplotlib colormap to use.
    
    Returns:
    - A nx3 numpy array where each column is an RGB color corresponding to an element in input_array.
    """
    if min_val is None:
        min_val = np.min(input_array)
    if max_val is None:
        max_val = np.max(input_array)

    # Normalize input_array to the range [0, 1]
    if max_val == min_val:
        value_range = 1.0
    else:
        value_range = max_val - min_val
    normalized_array = (input_array - min_val) / value_range
    
    # Get the colormap
    cmap = plt.get_cmap(colormap)
    
    # Map normalized data to colors
    colors = cmap(normalized_array)[:, :3]
        
    return colors

def gen_box(side_length:float, num_pts:int):
    """
    Generate points for a square in the XOZ plane, centered at (0, 0, 0), with a side length of 2.5
    The square will be parallel to the XOZ plane, so all x coordinates will be 0.
    
    Args:
        side_length (float): The length of the sides of the square.
        num_pts (int): The number of points to sample along one side of the square.
    
    Returns:
        box_pts (np.ndarray): A 2D array of shape (num_pts, 3) containing the coordinates of 
            the points. The points are in the format [[x1, y1, z1], [x2, y2, z2], ...].
    """

    # Number of points to sample along one side - for a total of 25 points, we need a 5x5 grid
    points_per_side = int(np.sqrt(num_pts))

    # Generating coordinates for X and Z
    x = np.linspace(-side_length / 2, side_length / 2, points_per_side)
    z = np.linspace(-side_length / 2, side_length / 2, points_per_side)

    # Meshgrid for X and Z coordinates
    X, Z = np.meshgrid(x, z)

    # Flatten X and Z arrays 
    X_flat = X.flatten()
    Z_flat = Z.flatten()
    # Create a corresponding Y array filled with zeros
    Y_flat = np.zeros(X_flat.shape)

    # Combine X, Y, Z into a single array of 3D points
    return np.vstack((X_flat, Y_flat, Z_flat)).T

def gen_trans_box(side_length, num_pts, transform) -> o3d.geometry.PointCloud:
    """
    Create a transformed box point cloud.

    Args:
        side_length (float): The length of the sides of the box.
        num_pts (int): The number of points to sample along one side of the box.
        transform (np.ndarray): A 4x4 transformation matrix to apply to the box.
    
    Returns:
        trans_box_pcd (o3d.geometry.PointCloud): The transformed box point cloud.
    """
    box_pts = gen_box(side_length, num_pts)
    trans_box_pcd = o3d.geometry.PointCloud()
    trans_box_pcd.points = o3d.utility.Vector3dVector(box_pts)

    # Rotate matrix around z axis by 90 degrees
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    trans_box_pcd.rotate(R)
    trans_box_pcd.transform(transform)
    return trans_box_pcd

def create_ball(radius=0.01, color=[1, 0, 0], center=None):
    """
    Create an Open3D sphere mesh with a specified radius and color.
    
    Args:
        radius (float): The radius of the sphere.
        color (list): A list of three floats representing the RGB color of the sphere.
        center (list): A list of three floats representing the center of the sphere.
            
    Returns:
        o3d.geometry.TriangleMesh: The created sphere mesh.
    """
    ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    ball.compute_vertex_normals()
    ball.paint_uniform_color(color)
    if center is not None:
        ball.translate(center)
    return ball

def create_motion_lines(prev_pts, curr_pts, return_pcd=False):
    """
    Create a line set to visualize the motion between two sets of points.
    
    Args:

        prev_pts (np.ndarray): shape (N, 3), the previous points.
        curr_pts (np.ndarray): shape (N, 3), the current points.
        return_pcd (bool): If True, return the point clouds along with the line set.
    
    Returns:
        o3d.geometry.LineSet: The created line set.
        tuple: If return_pcd is True, return the previous and current point clouds along with the line set.
    """
    assert(prev_pts.shape == curr_pts.shape)
    prev_pcd = o3d.geometry.PointCloud()
    prev_pcd.points = o3d.utility.Vector3dVector(prev_pts)
    prev_pcd.paint_uniform_color([0, 0, 1])

    curr_pcd = o3d.geometry.PointCloud()
    curr_pcd.points = o3d.utility.Vector3dVector(curr_pts)
    curr_pcd.paint_uniform_color([1, 0, 0])

    pcd_correspondence = [[i, i] for i in range(curr_pts.shape[0])]
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(prev_pcd, curr_pcd, pcd_correspondence)
    if return_pcd:
        return prev_pcd, curr_pcd, line_set
    else:
        return line_set

def create_graph_lines(points, edges):
    """ Visualize line set """
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    return line_set

def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def create_vector_arrow(end, origin=np.array([0, 0, 0]), scale=1, color=[0.707, 0.707, 0.0]):
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return(mesh)


def create_arrow_lst(p1_ary, p2_ary, **args):
    arrow_lst = []
    for p1, p2 in zip(p1_ary, p2_ary):
        if np.linalg.norm(p2-p1) > 0.01:
            arrow_lst.append(create_vector_arrow(p2, origin=p1, **args))
    return arrow_lst

def set_view_params(o3d_vis, view_params={}):
    ctr = o3d_vis.get_view_control()
    if "zoom" in view_params.keys():
        ctr.set_zoom(view_params["zoom"])
    if "front" in view_params.keys():
        ctr.set_front(view_params["front"])
    if "lookat" in view_params.keys():
        ctr.set_lookat(view_params["lookat"])
    if "up" in view_params.keys():
        ctr.set_up(view_params["up"])

def select_points(pcd, view_params=None):
    """
    Utility function to select points in a point cloud using Open3D's visualizer.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to visualize.
        view_params (dict): Optional dictionary containing view parameters for the visualizer.
            Keys can include 'zoom', 'front', 'lookat', and 'up'.
    Returns:
        list: A list of indices of the selected points.
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    if view_params is not None:
        set_view_params(vis, view_params)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()


def create_cylinder_between_two_points(p1, p2, radius, resolution=15, split=150):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Compute the vector from p1 to p2
    direction = p2 - p1
    # length = np.linalg.norm(direction)
    length = np.sqrt(np.sum(direction**2))
    direction = direction / length

    # Create a cylinder centered at the origin
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, 
                                                         resolution=resolution, split=split)

    Rz, Ry = calculate_zy_rotation_for_arrow(direction)

    cylinder.rotate(Ry, center=np.array([0, 0, 0]))
    cylinder.rotate(Rz, center=np.array([0, 0, 0]))

    # Translate the cylinder
    # cylinder.translate(np.array([0, 0, length / 2]))
    # cylinder.translate(p1)
    cylinder.translate((p1 + p2) / 2, relative=False)

    return cylinder
