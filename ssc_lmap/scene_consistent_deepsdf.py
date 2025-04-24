import numpy as np
import open3d as o3d
import torch
import pypose as pp
import time
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from .HortiMapping.deepsdf.deep_sdf import config_decoder, load_latent_vectors
from .HortiMapping.wild_completion.mesher import MeshExtractor

from .vis_utils import create_motion_lines


from dataclasses import dataclass

@dataclass
class FitWeightConfig:
    """
    Configuration for loss weights used in plant point cloud fitting.

    Attributes:
        weight_surface (float): Weight for surface alignment loss.
        weight_negative (float): Weight for penalizing points in free space.
        weight_peduncle (float): Weight for peduncle connection loss.
        weight_regularization (float): Weight for regularization term.
        peduncle_tolerance (float): Distance threshold for peduncle connection.
        weight_neg_com_branch (float): Weight for penalizing negative branch correspondences.
    """
    weight_surface: float
    weight_negative: float
    weight_peduncle: float
    weight_regularization: float
    peduncle_tolerance: float
    weight_neg_com_branch: float

    def to_dict(self) -> dict:
        """Convert the weight configuration to a dictionary format."""
        return {
            'weight_surface': self.weight_surface,
            'weight_negative': self.weight_negative,
            'weight_peduncle': self.weight_peduncle,
            'weight_regularization': self.weight_regularization,
            'peduncle_tolerance': self.peduncle_tolerance,
            'weight_neg_com_branch': self.weight_neg_com_branch
        }

@dataclass
class FruitCompletionConfig:
    """
    Configuration for fitting fruit point cloud using scene-consistent DeepSDF.

    Attributes:
        deepsdf_dir (str): Directory containing the DeepSDF model.
        weights (FitWeightConfig): Loss weight configuration.
        opt_mesh_res (float): Resolution of the mesh used during optimization.
        fruit_min_size (float): Minimum allowable size for the estimated fruit.
        num_correspondence (int): Number of point correspondences for alignment.
        NUM_ITERS (int): Number of optimization iterations.
        lr (float): Learning rate used during optimization.
        output_mesh_res (float): Resolution of the final output mesh.
    """
    deepsdf_dir: str
    weights: FitWeightConfig
    opt_mesh_res: float
    fruit_min_size: float
    num_correspondence: int
    NUM_ITERS: int
    lr: float
    output_mesh_res: float



def decode_sdf(decoder: torch.nn.Module, lat_vec: torch.Tensor, 
               x: torch.Tensor, max_batch=64**3, with_grad=False) -> torch.Tensor:
    """
    Decode the SDF values given the latent code and query points
    
    Args:
        decoder (torch.nn.Module): The DeepSDF decoder.
        lat_vec (torch.Tensor): shape (code_len,), the latent code.
        x (torch.Tensor): shape (N, 3), the query positions.
        max_batch (int): The maximum batch size for decoding.
        with_grad (bool): Whether to compute gradients.
    
    Returns:
        sdf_values (torch.Tensor): shape (N,), the SDF values for the query positions.
    """

    num_samples = x.shape[0]

    head = 0

    if with_grad:
        # get sdf values given query points
        sdf_values_chunks = []
        while head < num_samples:
            x_subset = x[head : min(head + max_batch, num_samples), 0:3].cuda()

            latent_repeat = lat_vec.expand(x_subset.shape[0], -1)
            fp_inputs = torch.cat([latent_repeat, x_subset], dim=-1)
            sdf_values = decoder(fp_inputs).squeeze()

            sdf_values_chunks.append(sdf_values)
            head += max_batch
    else:
        with torch.no_grad():
            # get sdf values given query points
            sdf_values_chunks = []
            while head < num_samples:
                x_subset = x[head : min(head + max_batch, num_samples), 0:3].cuda()

                latent_repeat = lat_vec.expand(x_subset.shape[0], -1)
                fp_inputs = torch.cat([latent_repeat, x_subset], dim=-1)
                sdf_values = decoder(fp_inputs).squeeze()

                sdf_values_chunks.append(sdf_values)
                head += max_batch

    sdf_values = torch.cat(sdf_values_chunks, 0).cuda()
    return sdf_values


class SceneConsistentDeepSDF:
    """
    DeepSDF shape completion with scene-consistent constraints.
    
    Attributes:
        completion_config (FruitCompletionConfig): Configuration for fruit completion.
        checkpoint (str): The checkpoint name, default is 'latest'.
        device (torch.device): The device to run the model on.
        decoder (torch.nn.Module): The DeepSDF decoder.
        init_latent (torch.Tensor): The initial latent code.
        code_len (int): The length of the latent code.
    """
    def __init__(self, completion_config: FruitCompletionConfig, checkpoint:str='latest'):
        self.config = completion_config
        self.checkpoint = checkpoint
        deepsdf_dir = completion_config.deepsdf_dir

        # load deep sdf decoder and init latent code
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decoder:torch.nn.Module = config_decoder(deepsdf_dir, self.checkpoint)
        self.decoder.to(self.device)

        latents_train = load_latent_vectors(deepsdf_dir, checkpoint).to(self.device)
        self.init_latent = torch.mean(latents_train, 0) # the mean latent code for training data
        self.code_len = self.init_latent.shape[0]

        print("INFO: DeepSDF model loaded")
    
    def get_fruit_center(self, fruit_pcd, cam_center:np.ndarray, fruit_min_size:float):
        """
        Get the center of the fruit point cloud.
        This function considers the fruit point cloud is partially occluded
        and move the center by fruit_min_size along the camera center direction.
        
        Args:
            fruit_pcd (o3d.geometry.PointCloud): The fruit point cloud.
            cam_center (np.ndarray): shape (3,), the camera center.
            fruit_min_size (float): The minimum size of the fruit.
        
        Returns:
            fruit_center (np.ndarray): shape (3,), the center of the fruit point cloud.
        """
        fruit_center = np.asarray(fruit_pcd.get_center())
        move_direction = fruit_center - cam_center
        move_direction /= np.linalg.norm(move_direction)
        fruit_center = fruit_center + fruit_min_size * move_direction
        return fruit_center
    
    def fit_plant_pcd(self, fruit_pcd, branch_pcd, original_branch_pcd, leaf_pcd, 
                      cam_center:np.ndarray, free_space_pcd : o3d.geometry.PointCloud, 
                      opt_mesh_res:int=None, fruit_min_size:float=None, 
                      num_correspondence:int=None, weight_dict:dict=None,
                      NUM_ITERS:int=None, lr:float=None, output_mesh_res:int=None, 
                      verbose=False, vis=False):
        """
        Fit deepSDF model to the plant point cloud
        Using fruit, `completed` branch, leaf, free space point cloud
        The camera center is used to move the fruit center along the camera center direction 
        
        Args:
            fruit_pcd (o3d.geometry.PointCloud): The fruit point cloud.
            branch_pcd (o3d.geometry.PointCloud): Completed branch point cloud.
            original_branch_pcd (o3d.geometry.PointCloud): Original partial branch point cloud.
            leaf_pcd (o3d.geometry.PointCloud): Leaf point cloud.
            cam_center (np.ndarray): shape (3,), the camera center.
            free_space_pcd (o3d.geometry.PointCloud): Free space point cloud.
            opt_mesh_res (int): Resolution of the mesh used during optimization.
            fruit_min_size (float): Minimum size of the fruit, used to move the fruit center.
            num_correspondence (int): Number of correspondence points for the peduncle.
            weight_dict (dict): A dictionary containing the weights for the loss functions.
            NUM_ITERS (int): Number of iterations for optimization.
            lr (float): Learning rate for optimization.
            output_mesh_res (float): The resolution of the output mesh.
            verbose (bool): Whether to print the optimization process.
            vis (bool): Whether to visualize the optimization process.

        Returns:    
            dict: A dictionary containing the following items:
                - final_shape_mesh (o3d.geometry.TriangleMesh): The final shape mesh.
                - translation (np.ndarray): The translation vector.
                - rotation (np.ndarray): The rotation matrix.
                - scale (float): The scale factor.
                - latent (np.ndarray): The latent code.
                - time (float): The total time taken for optimization.

        """

        # Compute the center of the fruit moved along the camera center direction
        fruit_color_mean = np.mean(np.asarray(fruit_pcd.colors), axis=0)
        fruit_center = self.get_fruit_center(fruit_pcd, cam_center, self.config.fruit_min_size)

        # translation
        t = torch.tensor(fruit_center, dtype=torch.float32, requires_grad=True)
        # rotation
        r = pp.identity_so3(dtype=torch.float32)
        r = r.clone().detach().requires_grad_(True)
        # scale
        s = torch.ones(1, dtype=torch.float32, requires_grad=True)
        # latent code of sdf
        latent = torch.tensor(self.init_latent, dtype=torch.float32, requires_grad=True)

        if verbose:
            print('INFO: init sim3 params')
            print ('t:', t.detach().numpy())
            print ('r:', pp.matrix(r).detach().numpy())
            print ('s:', s.item())

        # extract the init mesh from the init latent code
        object_radius_max_m = 0.08

        mesh_extractor = MeshExtractor(self.decoder, code_len=self.code_len, 
                                       voxels_dim=self.config.opt_mesh_res, 
                                       cube_radius=object_radius_max_m) # mc res: 0.2/40 ~ 5mm
        # init_shape_mesh = mesh_extractor.complete_mesh(init_latent, torch.eye(4), fruit_color_mean.tolist())

        # init surface points for optimization
        init_fruit_pts = np.asarray(fruit_pcd.points)
        # init negative points for optimization
        init_negative_pts = np.array((leaf_pcd + original_branch_pcd + free_space_pcd).points)
        # init peduncle corresponding points for optimization, select num_correspondence peduncle points from the max z points on init shape pcd
        branch_pts = np.asarray(branch_pcd.points)
        branch_knn = NearestNeighbors(n_neighbors=1).fit(branch_pts)
        # init negative complete points for optimization
        init_completed_branch_pts = np.array(branch_pcd.points)

        optimizer = torch.optim.Adam([t, r, s, latent], lr=self.config.lr)
        # optimizer = torch.optim.Adam([t, r, s], lr=lr)

        loss_lst = []
        
        weights = self.config.weights

        time_start = time.time()
        for iter in range(self.config.NUM_ITERS):
            ###### optimize the latent with t, r, s with losses ######
            iter_start_time = time.time()
            
            # move observation to the current fruit coordinate
            current_fruit_tsr = torch.tensor(init_fruit_pts, dtype=torch.float32)
            # transform using t
            current_fruit_tsr = current_fruit_tsr - t
            # transform using r
            current_fruit_tsr = pp.Exp(r) @ current_fruit_tsr
            # rescale using s
            current_fruit_tsr = 1.0 / s * current_fruit_tsr
            # decode the current latent code using the current fruit points
            decoded_fruit_sdf = decode_sdf(self.decoder, latent, current_fruit_tsr, with_grad=True)
            # compute loss_surface using the decoded fruit sdf by l2 loss
            loss_surface = torch.sum(decoded_fruit_sdf ** 2)
            loss_surface /= len(decoded_fruit_sdf)
            # print ('loss_surface:', loss_surface)

            # move negative points to the current fruit coordinate
            current_negative_tsr = torch.tensor(init_negative_pts, dtype=torch.float32)
            # transform using t
            current_negative_tsr = current_negative_tsr - t
            # transform using r
            current_negative_tsr = pp.Exp(r) @ current_negative_tsr
            # rescale using s
            current_negative_tsr = 1.0 / s * current_negative_tsr
            # decode the current latent code using the current negative points
            decoded_negative_sdf = decode_sdf(self.decoder, latent, current_negative_tsr, with_grad=True)
            decoded_negative_sdf_neg = decoded_negative_sdf[ decoded_negative_sdf < 0 ]
            decoded_negative_sdf_neg_num = len(decoded_negative_sdf_neg)
            # compute loss_negative using the decoded negative sdf by our relu function
            if decoded_negative_sdf_neg_num == 0:
                loss_negative = torch.tensor(0.0)
            else:
                loss_negative = torch.sum(torch.relu(-decoded_negative_sdf) ** 2)
                loss_negative /= decoded_negative_sdf_neg_num
            # print ('loss_negative:', loss_negative)

            # move neg_com_branch_tsr points to the current fruit coordinate
            current_neg_com_branch_tsr = torch.tensor(init_completed_branch_pts, dtype=torch.float32)
            # transform using t
            current_neg_com_branch_tsr = current_neg_com_branch_tsr - t
            # transform using r
            current_neg_com_branch_tsr = pp.Exp(r) @ current_neg_com_branch_tsr
            # rescale using s
            current_neg_com_branch_tsr = 1.0 / s * current_neg_com_branch_tsr
            # decode the current latent code using the current negative points
            decoded_neg_com_branch_sdf = decode_sdf(self.decoder, latent, current_neg_com_branch_tsr, with_grad=True)
            decoded_neg_com_branch_sdf_neg = decoded_neg_com_branch_sdf[ decoded_neg_com_branch_sdf < 0 ]
            decoded_neg_com_branch_sdf_neg_num = len(decoded_neg_com_branch_sdf_neg)
            # compute loss_negative using the decoded negative sdf by our relu function
            if decoded_neg_com_branch_sdf_neg_num == 0:
                loss_neg_com_branch = torch.tensor(0.0)
            else:
                loss_neg_com_branch = torch.sum(torch.relu(-decoded_neg_com_branch_sdf) ** 2)
                loss_neg_com_branch /= decoded_neg_com_branch_sdf_neg_num
            # print ('loss_neg_com_branch:', loss_neg_com_branch)

            # extract the now shape mesh
            current_shape_mesh = mesh_extractor.complete_mesh(latent, torch.eye(4), 
                                                              [fruit_color_mean[0], fruit_color_mean[1], fruit_color_mean[2]])
            current_peduncle_pts = np.asarray(current_shape_mesh.vertices)
            # select num_correspondence peduncle points from the max z points
            current_peduncle_pts = current_peduncle_pts[np.argsort(current_peduncle_pts[:, 2])[-self.config.num_correspondence:], :]
            # move peduncle points to the world coordinate
            current_peduncle_tsr = torch.tensor(current_peduncle_pts, dtype=torch.float32)
            # rescale using s
            current_peduncle_tsr = s * current_peduncle_tsr
            # transform using r
            current_peduncle_tsr = pp.Exp(r) @ current_peduncle_tsr
            # transform using t
            current_peduncle_tsr = current_peduncle_tsr + t
            # find the corresponding world peduncle points using kd-tree
            current_peduncle_pcd = o3d.geometry.PointCloud()
            current_peduncle_pcd.points = o3d.utility.Vector3dVector(current_peduncle_tsr.detach().numpy())
            current_peduncle_pts = np.asarray(current_peduncle_pcd.points)
            # find the corresponding branch points
            branch_eud, branch_idx = branch_knn.kneighbors(current_peduncle_pts)
            match_branch_pts = branch_pts[branch_idx.flatten(), :]
            match_branch_tsr = torch.tensor(match_branch_pts, dtype=torch.float32)
            # compute loss_peduncle using the current peduncle points by l2 loss
            loss_peduncle = torch.sum((current_peduncle_tsr - match_branch_tsr) ** 2)
            loss_peduncle /= len(match_branch_tsr)
            if loss_peduncle < weights.peduncle_tolerance:
                loss_peduncle = torch.tensor(0.0)
            # print ('loss_peduncle:', loss_peduncle)

            # compute loss_regularization using the latent code by l2 loss
            loss_regularization = torch.sum(latent ** 2)
            # print ('loss_regularization:', loss_regularization)

            # compute total loss
            # loss = loss_surface + loss_negative + loss_regularization + loss_peduncle
            loss = weights.weight_surface * loss_surface \
                + weights.weight_negative * loss_negative \
                + weights.weight_peduncle * loss_peduncle \
                + weights.weight_regularization * loss_regularization \
                + weights.weight_neg_com_branch * loss_neg_com_branch
            print ('total loss:', loss)

            # if vis and (iter+1) % 50 == 0:
            if vis:
                # visualize the current shape mesh and the peduncle points
                lines_peduncle_and_branch = create_motion_lines(current_peduncle_pts, match_branch_pts, return_pcd=False)
                current_shape_mesh.scale(s.item(), center = [0, 0, 0])
                rot_mat = pp.matrix(r).detach().numpy()
                transform_mat = np.eye(4)
                transform_mat[:3, :3] = rot_mat
                current_shape_mesh.transform(transform_mat)
                current_shape_mesh.translate(t.detach().numpy())
                
                # o3d.io.write_triangle_mesh(f'out_data/test_obj_100_complete_process/all_loss/current_shape_mesh_{iter:03d}.ply', 
                #                            current_shape_mesh)
                # o3d.visualization.draw_geometries([fruit_pcd, branch_pcd, leaf_pcd, free_space_pcd, current_shape_mesh, lines_peduncle_and_branch])
                # o3d.visualization.draw_geometries([current_shape_mesh, current_peduncle_pcd, branch_pcd, lines_peduncle_and_branch])
                # o3d.visualization.draw_geometries([current_shape_mesh, fruit_pcd, leaf_pcd, original_branch_pcd])

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss_lst.append(loss.item())
            
            # iter_end_time = time.time()
            # print ('t:', t.detach().numpy())
            # print ('r:', pp.matrix(r).detach().numpy())
            # print ('s:', s.item())
            # # print ('latent:', latent.detach().numpy())
            # print ('iter time:', iter_end_time - iter_start_time)

        time_end = time.time()
        print ('total time:', time_end - time_start)

        if vis:
            plt.plot(loss_lst)
            plt.show()

        # convert to 4x4 matrix
        t_np = t.detach().numpy()
        r_np = pp.matrix(r).detach().numpy()
        s_np = s.item()
        
        if verbose:
            print ('t:', t_np)
            print ('r:', r_np)
            print ('s:', s_np)

        # visualize the result
        final_mesh_extractor = MeshExtractor(self.decoder, code_len=self.code_len, 
                                             voxels_dim=self.config.output_mesh_res, 
                                             cube_radius=object_radius_max_m) # mc res: 0.2/40 ~ 5mm
        
        fruit_color_mean_list = [fruit_color_mean[0], fruit_color_mean[1], fruit_color_mean[2]]
        final_shape_mesh = final_mesh_extractor.complete_mesh(latent, torch.eye(4), fruit_color_mean_list)
        final_shape_mesh.scale(s_np, center = [0, 0, 0])
        transform_mat = np.eye(4)
        transform_mat[:3, :3] = r_np
        final_shape_mesh.transform(transform_mat)
        final_shape_mesh.translate(t_np)

        # Visualize the final shape mesh
        if verbose:
            o3d.visualization.draw_geometries([fruit_pcd, branch_pcd, original_branch_pcd, 
                                               leaf_pcd, free_space_pcd, final_shape_mesh])

        self.t = t
        self.r = r
        self.s = s
        self.latent = latent

        return {
            'final_shape_mesh': final_shape_mesh, 
            'translation': self.t.detach().cpu().numpy(),
            'rotation': pp.matrix(self.r).detach().cpu().numpy(),
            'scale': self.s.detach().cpu().numpy(),
            'latent': self.latent.detach().cpu().numpy(),
            'time': time_end - time_start
        }
    
    def points_sdf_value(self, points):
        """
        Compute the sdf value of the points given the latent code
        
        Args:
            points (np.ndarray): shape (N, 3), the query points.
        
        Returns:
            sdf_values (np.ndarray): shape (N,), the sdf values for the query points.
        """
        points_tsr = torch.tensor(points, dtype=torch.float32)
        points_tsr = points_tsr - self.t
        points_tsr = pp.Exp(self.r) @ points_tsr
        points_tsr = 1.0 / self.s * points_tsr
        sdf_values = decode_sdf(self.decoder, self.latent, points_tsr, with_grad=False)
        return sdf_values.detach().cpu().numpy()
    
