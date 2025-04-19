import numpy as np
import time
import open3d as o3d
import matplotlib.pyplot as plt
import pypose as pp
import scipy.sparse as scisp
import sklearn.neighbors as skn

import torch
from torch import nn

from .vis_utils import scalars_to_colors

from .pts_utils import connect_points, connect_leaf2branch, select_close_points, assign_edge_weights

def prepare_sim_obj(box_start_pcd, branch_pcd, leaf_pcd, 
                    close_branch_distance=0.05, 
                    voxel_size=0.003, nn_radius=0.01, 
                    leaf2branch_weight=100, verbose=False):
    """
    Prepare simulator object for the leaf and branch point clouds.
    
    The creates embedded deformation graph for the following points: 
    box point on the gripper of the robot, branch point cloud, and leaf point cloud.
    
    Args:
        box_start_pcd (o3d.geometry.PointCloud): Representing grasp points a rectangle box.
        branch_pcd (o3d.geometry.PointCloud): Point cloud of the branch.
        leaf_pcd (o3d.geometry.PointCloud): Point cloud of the leaf.
        close_branch_distance (float): Distance threshold to collect leaf points close to the branch.
        voxel_size (float): Voxel size for downsampling the point clouds.
        nn_radius (float): Radius for nearest neighbor search on the leaf point cloud.
        leaf2branch_weight (float): Weight for the edges connecting leaf points to branch points.
        verbose (bool): If True, visualize the point clouds and edges.
    
    Returns:
        all_vis_pcd (o3d.geometry.PointCloud): Point cloud containing all points for visualization. 
        all_sim_pts (np.ndarray): shape (N, 3) containing all points for simulation.
        handle_idx (np.ndarray): shape (N,) containing indices of the handle points.
        connect_ary (np.ndarray): shape (M, 2) containing the edges connecting points in the graph.
        edge_weights (np.ndarray): shape (M,) containing weights for the edges in the graph.
    """
    all_vis_pts = np.vstack([np.array(branch_pcd.points), np.array(leaf_pcd.points)])

    all_vis_pcd = o3d.geometry.PointCloud()
    all_vis_pcd.points = o3d.utility.Vector3dVector(all_vis_pts)
    all_vis_pcd.colors = o3d.utility.Vector3dVector(np.vstack([np.array(branch_pcd.colors), np.array(leaf_pcd.colors)]))

    close_branch_pcd = select_close_points(leaf_pcd, branch_pcd, close_branch_distance)

    # Concatenate downsampled point clouds
    close_branch_pcd = close_branch_pcd.voxel_down_sample(voxel_size=voxel_size)
    leaf_pcd = leaf_pcd.voxel_down_sample(voxel_size=voxel_size)
    all_sim_pts = np.vstack([np.array(box_start_pcd.points), np.array(close_branch_pcd.points), np.array(leaf_pcd.points)])
    # if verbose:
    #     o3d.visualization.draw_geometries([close_branch_pcd, leaf_pcd])

    handle_idx = np.arange(len(box_start_pcd.points) + len(close_branch_pcd.points))

    connect_ary = connect_points(all_sim_pts, nn_radius)

    leaf2branch_edges = connect_leaf2branch(np.array(leaf_pcd.points), np.array(close_branch_pcd.points), nn_radius*2.0)
    leaf2branch_edges += len(box_start_pcd.points)
    leaf2branch_edges[:, 0] += len(close_branch_pcd.points)

    # Visualize the lineset that connect leaf_pts and branch_pts with edges above
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_sim_pts)
    line_set.lines = o3d.utility.Vector2iVector(leaf2branch_edges)
    if verbose:
        o3d.visualization.draw_geometries([line_set, leaf_pcd, branch_pcd])

    print('number of handle points:', len(handle_idx))
    # Assign prior edge weights based on the relative position of the points
    # 3.0 for points close to the center, 2.0 for points between centers and edges, 1.0 for points on edges
    edge_weights = assign_edge_weights(leaf_pcd, all_sim_pts, connect_ary, w1=3.0, w2=2.0, w3=1.0)
    edge_weights = np.array(edge_weights)

    edge2weight_dict = {}
    for edge, weight in zip(connect_ary, edge_weights):
        edge2weight_dict[(edge[0], edge[1])] = weight
    
    for edge in leaf2branch_edges:
        edge2weight_dict[(edge[0], edge[1])] = leaf2branch_weight
    
    connect_ary = np.array(list(edge2weight_dict.keys()))
    edge_weights = np.array(list(edge2weight_dict.values()))

    # connect_ary = np.concatenate([connect_ary, leaf2branch_edges], axis=0)
    # edge_weights = np.concatenate([edge_weights, leaf2branch_weight*np.ones(len(leaf2branch_edges))], axis=0)
    return all_vis_pcd, all_sim_pts, handle_idx, connect_ary, edge_weights


class DeformState(nn.Module):
    """
    DeformState class for representing the deformation state of a point cloud.
    
    This class is attached to a specific set of rest points and describes
    the deformation of each point as (N, 3) tensor, where N is the number of points.
    This is optional rotation matrix associated with each point, which is used for
    ARAP deformation to compute the rotation of the edges.
    
    Attributes:
        num_pts (int): Number of points in the point cloud.
        delta_pts_tsr (torch.Tensor): shape (N, 3) tensor representing the deformation of each point.
        corotate (bool): If True, the deformation is corotated.
        rot_tsr (torch.Tensor): shape (N, 4) so3 tensor representing the rotation of each point.
    """

    def __init__(self, num_pts, corotate=False) -> None:
        super().__init__()
        self.num_pts = num_pts
        self.delta_pts_tsr = nn.Parameter(torch.zeros(self.num_pts, 3), 
                                          requires_grad=True)
        self.corotate = corotate
        if corotate:
            self.rot_tsr = nn.Parameter(pp.randn_so3(self.num_pts, sigma=1e-5, 
                                                     requires_grad=True))
    
    def forward(self, rest_pts_tsr: torch.Tensor, 
                handle_idx:np.ndarray, handle_pts_tsr: torch.Tensor):
        """
        Predicted the points deformation for the given rest points, 
        for handle points they will move directly to the handle points.
        
        Args:
            rest_pts_tsr (torch.Tensor): shape (N, 3) tensor representing the rest points.
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
            handle_pts_tsr (torch.Tensor): shape (N_h, 3) tensor representing the handle points.
        
        Returns:
            curr_pts_tsr (torch.Tensor): shape (N, 3) tensor representing the current points.
        """
        curr_pts_tsr = rest_pts_tsr + self.delta_pts_tsr
        curr_pts_tsr[handle_idx] = handle_pts_tsr
        return curr_pts_tsr
    
    def compute_edge_rest(self, rest_pts_tsr: torch.Tensor, edges_ary: np.ndarray):
        """
        Compute the rest edge difference for the given rest points and edges.
        
        Here edges are directed and points to edges_ary[:, 0] from edges_ary[:, 1].
        
        1. Compute the edge difference for the rest points.
        2. If corotate is True, apply the rotation to the edge difference.
        3. Return the edge difference.
        
        Args:
            rest_pts_tsr (torch.Tensor): shape (N, 3) tensor representing the rest points.
            edges_ary (np.ndarray): shape (M, 2) array representing the edges.
        
        Returns:
            edges_rest (torch.Tensor): shape (M, 3) tensor representing the rest edge difference.
        """
        edges_rest = rest_pts_tsr[edges_ary[:, 0], :] - rest_pts_tsr[edges_ary[:, 1], :]
        if self.corotate:
            edges_rest = pp.Exp(self.rot_tsr[edges_ary[:, 0], :]) @ edges_rest
        return edges_rest

    def compute_edge_diff(self, rest_pts_tsr: torch.Tensor, 
                          handle_idx:np.ndarray, handle_pts_tsr: torch.Tensor, 
                          edges_ary: np.ndarray):
        """
        Compute the edge difference for the given rest points, handle points and edges.
        
        Here edges are directed and points to edges_ary[:, 0] from edges_ary[:, 1].
        
        1. Compute the current points using the forward method.
        2. Compute the edge difference for the current points.
        
        Args:
            rest_pts_tsr (torch.Tensor): shape (N, 3) tensor representing the rest points.
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
            handle_pts_tsr (torch.Tensor): shape (N_h, 3) tensor representing the handle points.
            edges_ary (np.ndarray): shape (M, 2) array representing the edges.
        
        Returns:
            edges_diff (torch.Tensor): shape (M, 3) tensor representing the edge difference.
        """
        curr_pts_tsr = self.forward(rest_pts_tsr, handle_idx, handle_pts_tsr)
        edges_diff = curr_pts_tsr[edges_ary[:, 0], :] - curr_pts_tsr[edges_ary[:, 1], :]
        return edges_diff

class NodeGraph:
    """
    Embedded deformation graph that can simulate points deformation by minimizing the ARAP energy.
    
    This class is instantiated by a set of 3D points and a set of edges connecting the points.
    Each edge has a weight associated with it, which is used to compute the energy of the deformation.
    
    
    """
    def __init__(self, rest_pts:np.ndarray, edges:np.ndarray, 
                 edge_weights:np.ndarray=None, corotate=False, 
                 num_nns:int=10, dtype=torch.double, device='cpu') -> None:
        """
        Args:
            rest_pts (np.ndarray): shape (N, 3) array representing the rest points.
            edges (np.ndarray): shape (M, 2) array representing the edges.
            edge_weights (np.ndarray): shape (M,) array representing the edge weights.
            corotate (bool): If True, also compute per-vertex rotation.
            num_nns (int): Number of nearest neighbors for
                the nearest neighbor search to perform
            dtype (torch.dtype): Data type for the tensors.
            device (str): Device for the tensors.
        """

        self.tsr_params = {'dtype': dtype, 'device': device}

        self.rest_pts_tsr = torch.tensor(rest_pts, **self.tsr_params)
        self.edges_tsr = torch.tensor(edges, device=device, dtype=torch.long)

        self.num_pts = self.rest_pts_tsr.shape[0]
        self.num_edges = self.edges_tsr.shape[0]

        self.corotate = corotate
        self.deform_state = DeformState(self.num_pts, corotate=corotate)
        self.deform_state.to(**self.tsr_params)

        if edge_weights is None:
            self.edges_weight_tsr = torch.ones(self.num_edges, **self.tsr_params)
        else:
            self.edges_weight_tsr = torch.tensor(edge_weights, **self.tsr_params)

        self.node_knn = skn.NearestNeighbors(n_neighbors=10)
        self.num_nns = num_nns
        self.node_knn.fit(rest_pts)

        self.setup_graph_matrix()
    
    def reset_state(self):
        """
        Reset the deformation state of the graph to no deformation and no rotation.
        """
        self.deform_state = DeformState(self.num_pts, corotate=self.corotate)
        self.deform_state.to(**self.tsr_params)
    
    def setup_graph_matrix(self):
        """
        This function prepares the graph matrices for the `global` step in the ARAP deformation.
        
        1. Assemble the Laplace matrix for the graph.
        2. Compute the rhs_matrix, which is used to compute the edge difference.
        3. Compute the rest edge difference, which is used to compute the energy of the deformation.
        """
        num_edges = self.edges_tsr.shape[0]

        start_time = time.time()
        row_idx = torch.cat([self.edges_tsr[:, 0], self.edges_tsr[:, 1], self.edges_tsr[:, 0], self.edges_tsr[:, 1]])
        col_idx = torch.cat([self.edges_tsr[:, 0], self.edges_tsr[:, 1], self.edges_tsr[:, 1], self.edges_tsr[:, 0]])
        indices = torch.vstack([row_idx, col_idx])
        values = torch.cat([self.edges_weight_tsr, self.edges_weight_tsr, -self.edges_weight_tsr, -self.edges_weight_tsr])
        self.laplace_matrix = torch.sparse_coo_tensor(indices, values, (self.num_pts, self.num_pts), **self.tsr_params)
        self.laplace_matrix = self.laplace_matrix.to_dense()

        range_num_edges = torch.arange(num_edges, device=self.tsr_params['device'])
        pos_idx = torch.vstack([self.edges_tsr[:, 0], range_num_edges])
        neg_idx = torch.vstack([self.edges_tsr[:, 1], range_num_edges])
        indices = torch.cat([pos_idx, neg_idx], axis=1)
        values = torch.cat([self.edges_weight_tsr, -self.edges_weight_tsr])
        self.rhs_matrix = torch.sparse_coo_tensor(indices, values, (self.num_pts, num_edges), **self.tsr_params)
        self.rest_edge_diff = self.rest_pts_tsr[self.edges_tsr[:, 0], :] - self.rest_pts_tsr[self.edges_tsr[:, 1], :]
        print('assemble time:', time.time() - start_time)
    
    def enable_learn_weight(self):
        """
        NOTE: unused function, but can be used to enable learning the edge weights.
        """
        self.edges_weight_tsr.requires_grad = True
    
    def get_curr_pts(self, handle_idx, handle_pts_tsr: torch.Tensor) -> torch.Tensor:
        """
        Get the current points for the given handle points.
        
        Args:
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
            handle_pts_tsr (torch.Tensor): shape (N_h, 3) tensor representing the handle points.
        
        Returns:
            curr_pts_tsr (torch.Tensor): shape (N, 3) tensor representing the current points.
        """
        return self.deform_state.forward(self.rest_pts_tsr, handle_idx, handle_pts_tsr)
    
    def get_delta_pts(self) -> torch.Tensor:
        """
        Get reference to the delta points tensor in the deform state.
        
        Returns:
            delta_pts_tsr (torch.Tensor): shape (N, 3) tensor representing the deformation of each point.
        """
        return self.deform_state.delta_pts_tsr

    def energy(self, handle_idx, handle_pts_tsr: torch.Tensor) -> torch.Tensor:
        """
        Get a scalar energy value for the deformation of the graph.
        
        If we denote each point rest position as p_i and current location as p_i', 
        the energy is defined as:
        
            E = sum_{(i,j) in edges} w_{ij} ||p_i' - p_j' - R_i' @ (p_i - p_j)||_2^2
        
        where R_i' is the rotation matrix associated with point i.
        
        The energy is computed as the sum of the squared differences between the 
        current edge difference and the rest edge difference,
        weighted by the edge weights, the same as the ARAP energy.
        
        Args:
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
            handle_pts_tsr (torch.Tensor): shape (N_h, 3) tensor representing the handle points.
        
        Returns:
            energy (torch.Tensor): scalar tensor representing the energy of the deformation.
        """
        edges_diff = self.deform_state.compute_edge_diff(self.rest_pts_tsr, 
                                                         handle_idx, handle_pts_tsr, 
                                                         self.edges_tsr)
        edges_rest = self.deform_state.compute_edge_rest(self.rest_pts_tsr, self.edges_tsr)

        edges_delta = (edges_diff - edges_rest).pow(2).sum(dim=1)
        energy = (edges_delta * self.edges_weight_tsr).sum()
        return energy
    
    def set_handle_idx(self, handle_idx):
        """
        Set the handle points for the graph.
        
        This graph will prepare the linear system to solve during `global` step in the ARAP deformation.
        
        Args:
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
        """
        self.handle_idx = handle_idx
        self.free_idx = np.setdiff1d(np.arange(self.num_pts), handle_idx)

        start_time = time.time()
        W_ff_mat = self.laplace_matrix[self.free_idx, :][:, self.free_idx]
        self.laplace_inverse = torch.linalg.inv(W_ff_mat + 1e-5*torch.eye(len(self.free_idx), **self.tsr_params))
        print('inverse time:', time.time() - start_time)

    def set_handle_idx_and_pts(self, handle_idx, handle_pts: torch.Tensor):
        """
        This function sets the handle points and the corresponding target points.
        
        This function will prepare the linear system to solve during `global` step in the ARAP deformation.
        
        Args:
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
            handle_pts (torch.Tensor): shape (N_h, 3) tensor representing the handle points.
        """
        self.free_idx = np.setdiff1d(np.arange(self.num_pts), handle_idx)

        W_fh_mat = self.laplace_matrix[self.free_idx, :][:, handle_idx]
        self.rhs_bias = -W_fh_mat @ handle_pts
    
    def solve_linear(self):
        """
        Run `global` step in the ARAP deformation.
        
        Solves a linear system of equations to find the new positions of the free points.        
        """
        if not self.corotate:
            edge_diff = self.rest_edge_diff
        else:
            edge_diff = pp.Exp(self.deform_state.rot_tsr[self.edges_tsr[:, 0], :]) @ self.rest_edge_diff
        all_rhs = self.rhs_matrix @ edge_diff

        start_time = time.time()

        rhs = all_rhs[self.free_idx, :] + self.rhs_bias
        new_free_curr_pts = self.laplace_inverse @ rhs

        self.deform_state.delta_pts_tsr[self.free_idx, :] = new_free_curr_pts - self.rest_pts_tsr[self.free_idx, :] 
        # print('solve time:', time.time() - start_time)
    
    def solve_rotate(self, handle_idx, handle_pts_tsr: torch.Tensor):
        """
        Run `local` step in the ARAP deformation.
        
        The rotation for each vertex is computed using the Singular Value Decomposition (SVD) of the covariance matrix.
        
        Args:
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
            handle_pts_tsr (torch.Tensor): shape (N_h, 3) tensor representing the handle points.
        """
        rest_edge_diff = (self.rest_pts_tsr[self.edges_tsr[:, 0], :] - self.rest_pts_tsr[self.edges_tsr[:, 1], :])
        curr_pts_tsr = self.rest_pts_tsr + self.deform_state.delta_pts_tsr
        curr_pts_tsr[handle_idx] = handle_pts_tsr
        curr_edge_diff = (curr_pts_tsr[self.edges_tsr[:, 0], :] - curr_pts_tsr[self.edges_tsr[:, 1], :])
        edge_cov_tsr = torch.bmm(rest_edge_diff[:, :, None], curr_edge_diff[:, None, :])    
        cov_tsr = torch.zeros((self.num_pts, 3, 3), **self.tsr_params)
        cov_tsr.index_add_(0, self.edges_tsr[:, 0], self.edges_weight_tsr[:, None, None]*edge_cov_tsr)

        U, sig, VH = torch.linalg.svd(cov_tsr)

        UH = U.permute(0, 2, 1)
        V = VH.permute(0, 2, 1)

        R = V @ UH
        det_R = torch.linalg.det(R)
        flip_mask = det_R < 0
        UH[flip_mask, 2, :] *= -1
        R = V @ UH
        assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0, **self.tsr_params))

        so3_tsr = pp.mat2SO3(R)
        self.deform_state.rot_tsr.data = pp.Log(so3_tsr)

    def solve_global_local(self, handle_idx, handle_pts_tsr: torch.Tensor, num_iters=100, 
                           energy_converge_threshold=0.00001, verbose=False):
        """
        Solve the ARAP deformation using the global-local method.
        The global step solves a linear system of equations to find the new positions of the free points.
        The local step computes the rotation for each vertex using the Singular Value 
        Decomposition (SVD) of the covariance matrix.
        
        The optimization is performed at a maximum of num_iters iterations,
        or until the energy reduction is less than energy_converge_threshold.
        
        NOTE: this function changes deform_state.delta_pts_tsr and rot_tsr.
        
        Args:
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
            handle_pts_tsr (torch.Tensor): shape (N_h, 3) tensor representing the handle points.
            num_iters (int): Number of iterations to run the optimization.
            energy_converge_threshold (float): Threshold for convergence of the energy.
            verbose (bool): If True, print the energy at each iteration.
        
        Returns:
            energy_lst (list): List of energy values at each iteration.
        """
        self.set_handle_idx_and_pts(handle_idx, handle_pts_tsr)

        energy_lst = []
        for iter_id in range(num_iters):
            energy = self.energy(handle_idx, handle_pts_tsr)
            energy_initial = energy.item()
            energy_lst.append(energy_initial)

            self.solve_linear()
            energy_after_linear = self.energy(handle_idx, handle_pts_tsr)
            energy_after = energy_after_linear.item()

            if self.corotate:
                self.solve_rotate(handle_idx, handle_pts_tsr)   
                energy_after_rotate = self.energy(handle_idx, handle_pts_tsr)
                assert energy_after_rotate <= energy_after_linear, f"energy_after_rotate: {energy_after_rotate}, energy_after_linear: {energy_after_linear}"
                energy_after = energy_after_rotate.item()

            if verbose:
                print('energy_initial:', energy_initial, 'energy_after:', energy_after)

            if energy_initial - energy_after < energy_converge_threshold:
                print('converged at iter:', iter_id, 'with energy_converge_threshold:', energy_converge_threshold)
                break
        return energy_lst

    def get_pts_beta(self, pts:np.ndarray, rbf_sig=0.5, rbf_w_max=0.2, dist_max=0.1):
        """
        This function uses computes the RBF linear skinning weights for the given points.
        
        The RBF weights are computed using the nearest neighbors of the points in the graph.
        Here the number of points N_q can be much larger than number of points in the graph N. 
        
        Args:
            pts (np.ndarray): shape (N_q, 3) array representing the points to compute the weights for.
            rbf_sig (float): Standard deviation for the RBF kernel.
            rbf_w_max (float): Maximum weight for the RBF kernel.
            dist_max (float): Maximum distance for the RBF kernel.
        
        Returns:
            beta_tsr (scipy.sparse.csr_matrix): shape (N_q, N) sparse matrix representing the RBF weights.
        """
        num_pts = pts.shape[0]

        # pts shape: N x 3
        eud_ary, idx_ary = self.node_knn.kneighbors(pts)

        rbf_weights:np.ndarray = np.exp(-eud_ary/rbf_sig)
        rbf_weights[eud_ary > dist_max] = 0.0
        rbf_weights[rbf_weights < rbf_w_max] = 0.0

        rbf_weights /= rbf_weights.sum(axis=1, keepdims=True) + 1e-5

        row_idx = np.repeat(np.arange(num_pts), self.num_nns)
        col_idx = idx_ary.reshape(-1)
        data = rbf_weights.reshape(-1)

        beta_tsr = scisp.csr_matrix((data, (row_idx, col_idx)), 
                                    shape=(num_pts, self.num_pts))
        return beta_tsr


    def get_line_set(self):
        """
        A utility function to create open3d line set for the graph.
        
        The line set is created using the rest points and edges of the graph.
        The colors of the edges are based on the edge weights.
        The colorization is performed adaptively based on the max and min values
        in the edge weights.
        
        Returns:
            line_set (open3d.geometry.LineSet): Line set representing the graph.
        """
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.rest_pts_tsr.cpu().numpy())

        edges_lst = self.edges_tsr.cpu().numpy().tolist()

        line_set.lines = o3d.utility.Vector2iVector(edges_lst)

        w = self.edges_weight_tsr.cpu().numpy()
        colors = scalars_to_colors(w, colormap='YlOrRd')
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def get_pcd(self, handle_idx=None, handle_pts_tsr:torch.Tensor=None):
        """
        A utility function to create open3d point cloud for the graph.
        
        The point cloud is created using the current points of the graph.
        
        By default, all points are colored in black and the handle points are colored in red.
        
        Args:
            handle_idx (np.ndarray): shape (N_h,) array representing the indices of the handle points.
            handle_pts_tsr (torch.Tensor): shape (N_h, 3) tensor representing the handle points.
        
        Returns:
            pcd (open3d.geometry.PointCloud): Point cloud representing the graph.
        """
        if handle_idx is None:
            curr_pts_tsr = self.rest_pts_tsr.clone()
        else:
            curr_pts_tsr = self.get_curr_pts(handle_idx, handle_pts_tsr)
        
        curr_pts_ary = curr_pts_tsr.detach().cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(curr_pts_ary)
        
        pcd_colors = np.zeros_like(curr_pts_ary)
        if handle_idx is not None:
            pcd_colors[handle_idx] = [1.0, 0.0, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        return pcd

    def get_rot_frames(self, skip_rate = 100):
        """
        A utility function to create open3d coordinate frames for the graph.
        
        The coordinate frames are created using the current points and the rotation matrices
        associated with each point.
        
        Optionally only show 1/skip_rate of the frames.
        Args:
            skip_rate (int): Skip rate for the frames.
        
        Returns:
            coord_frame_lst (list): List of open3d coordinate frames representing the graph.

        """
        coord_frame_lst = []
        rot_tsr = self.deform_state.rot_tsr
        curr_pts_tsr = self.rest_pts_tsr + self.deform_state.delta_pts_tsr

        for pt_i in range(0, self.num_pts, skip_rate):
            ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            pt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            rot_mat:torch.Tensor = pp.Exp(rot_tsr[pt_i, :]) @ torch.eye(3)

            pt_frame.rotate(rot_mat.detach().numpy())
            pt_frame.translate(curr_pts_tsr[pt_i, :].detach().numpy())
            pt_frame.paint_uniform_color([0.2, 0.9, 0.7])

            ref_frame.translate(curr_pts_tsr[pt_i, :].detach().numpy())
            ref_frame.paint_uniform_color([0.7, 0.2, 0.9])
            coord_frame_lst.append(pt_frame)
            coord_frame_lst.append(ref_frame)
        return coord_frame_lst

if __name__ == '__main__':
    rest_pts = torch.zeros(5, 3)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [1, 0], [2, 1], [3, 2], [4, 3]])

    ng = NodeGraph(rest_pts, edges, corotate=True)

    handle_idx = torch.tensor([0, 3], dtype=torch.long)
    handle_pts = torch.tensor([[0.0, 0.3, -0.3], [0.0, 0.3, -0.3]], dtype=torch.double)

    optimizer = torch.optim.Adam(ng.deform_state.parameters(), lr=0.1)

    energy_lst = []
    for _ in range(100):
        energy = ng.energy(handle_idx, handle_pts)

        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        energy_lst.append(energy.item())
        print('loss:', energy.item())
    
    plt.plot(energy_lst)
    plt.show()
