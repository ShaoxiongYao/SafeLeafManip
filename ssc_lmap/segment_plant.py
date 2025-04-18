import cv2
import numpy as np
import supervision as sv
import open3d as o3d

import torch
import torchvision

try:
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print('INFO: segment_plant.py is not run in the correct environment')
    Model = None
    sam_model_registry = None
    SamPredictor = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CLASSES = ["branch", "green leaf", "fruit"]
# CLASSES = ["branch", "leaf", "pepper"]
CLASSES = ["branch", "leaf", "sweet pepper"]

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, 
            xyxy: np.ndarray, reduce_mode: str='max') -> np.ndarray:
    """"
    Utility wrapper function to segment the image with Segment Anything Model (SAM).
    
    Args:
        sam_predictor (SamPredictor): The SAM predictor object.
        image (np.ndarray): The input image to be segmented.
        xyxy (np.ndarray): shape (N, 4), each item is in (x_min, y_min, x_max, y_max) format.
        reduce_mode (str): The mode to reduce the masks. Options are 'max' or 'and'.
    
    Returns:
        masks (np.ndarray): The segmented masks for the input image.
    """
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        if reduce_mode == 'max':
            index = np.argmax(scores)
            result_masks.append(masks[index])
        elif reduce_mode == 'and':
            final_mask = np.logical_and.reduce(masks, axis=0)
            result_masks.append(final_mask)
    return np.array(result_masks)

def segment_within_boxes(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Utility wrapper function to segment the image with Segment Anything Model (SAM) within the given boxes.
    
    TODO: this function duplicates the `segment` function, to be removed in the future.
    
    Args:
        sam_predictor (SamPredictor): The SAM predictor object.
        image (np.ndarray): The input image to be segmented.
        xyxy (np.ndarray): shape (N, 4), each item is in (x_min, y_min, x_max, y_max) format.
    
    Returns:
        masks (np.ndarray): The segmented masks for the input image.
    
    """
    result_masks = []
    for box in xyxy:
        box = box.astype(int)
        box_image = image.copy()
        box_mask = np.zeros_like(image)
        cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
        box_image = cv2.bitwise_and(box_image, box_mask)
        sam_predictor.set_image(box_image)
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        for mask in masks:
            cv2.imshow('mask', mask.astype(np.uint8) * 255)
            cv2.waitKey(0)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


class GroundedSAM:
    """
    A class as a wrapper for GroundingDINO and Segment Anything Model (SAM).
    
    Borrowed from: https://github.com/IDEA-Research/Grounded-Segment-Anything
    
    Attributes:
        device (torch.device): The device to run the model on (CPU or GPU).
        grounding_dino_model (Model): The GroundingDINO model.
        reduce_segment_mode (str): The mode to reduce the masks. Options are 'max' or 'and'.
        sam (sam_model_registry): The SAM model.
        sam_predictor (SamPredictor): The SAM predictor.
    """
    def __init__(self, GROUNDING_DINO_CONFIG_PATH="data/GroundingDINO_SwinT_OGC.py", 
                 GROUNDING_DINO_CHECKPOINT_PATH="data/groundingdino_swint_ogc.pth", 
                 SAM_ENCODER_VERSION="vit_h", SAM_CHECKPOINT_PATH="data/sam_vit_h_4b8939.pth",
                 reduce_segment_mode='max', device=None) -> None:
        
        self.device = device or DEVICE

        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, 
                                          model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

        self.reduce_segment_mode = reduce_segment_mode

        # Building SAM Model and SAM Predictor
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def segment(self, image: np.ndarray, classes, box_threshold: float=0.25, 
                text_threshold: float=0.25, nms_threshold: float=0.8, verbose=False) -> np.ndarray:
        """
        Segment the image with GroundingDINO and SAM.
        
        First detect bounding boxes with GroundingDINO and classes as prompts.
        Then use NMS to filter the boxes. Finally, for each box detected by GroundingDINO
        run SAM to segment the image within the boxes. 
        
        TODO: refactor the return as a dataclass? The bbox_image and mask_image are not
        necessary to be returned in the same function.
        
        Args:
            image (np.ndarray): The input image to be segmented.
            classes (list): The classes to be detected.
            box_threshold (float): The threshold for the bounding boxes.
            text_threshold (float): The threshold for the text.
            nms_threshold (float): The threshold for NMS.
            verbose (bool): Whether to print verbose information.
        
        Returns:
            dict: A dictionary containing the following keys:
                - 'detections': The detected bounding boxes and masks.
                - 'bbox_image': The annotated image with bounding boxes.
                - 'mask_image': The annotated image with masks.
        """

        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image, classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # NMS post process
        if verbose:
            print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        if verbose:
            print(f"After NMS: {len(detections.xyxy)} boxes")
        
        # annotate image with detections after NMS
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ in detections]
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy, 
            reduce_mode=self.reduce_segment_mode
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        return {
            'detections': detections,
            'bbox_image': annotated_frame, 
            'mask_image': annotated_image,
        }
    
    def sam_segment(self, image, bbox):
        """
        Run SAM to segment the image within the given bounding boxes.
        
        Args:
            image (np.ndarray): The input image to be segmented.
            bbox (np.ndarray): shape (N, 4), each item is in (x_min, y_min, x_max, y_max) format.
        
        Returns:
            masks (np.ndarray): The segmented masks for the input image.
        """
        return segment(self.sam_predictor, image, bbox, reduce_mode=self.reduce_segment_mode)

    def get_segment_pcd(self, colors:np.ndarray, points:np.ndarray, classes, depth_min:float, depth_max:float, 
                        box_threshold:float, text_threshold:float, nms_threshold:float, verbose=False):
        """
        Get the segmented point cloud from the image and point cloud.
        
        A few notes:
        1. The colors is assumed to be of BGR cv2 format.
        2. The points is assumed to be in the (H, W, 3) format, 
        which are projected (x, y, z) points.
        
        TODO: refactor the return as a dataclass? Only pcd_lst is needed.
        
        Args:
            colors (np.ndarray): shape (H, W, 3), the color image in BGR format.
            points (np.ndarray): shape (H, W, 3), the point cloud in (x, y, z) format.
            classes (list): The classes to be detected.
            depth_min (float): The minimum depth to be considered.
            depth_max (float): The maximum depth to be considered.
            box_threshold (float): The threshold for the bounding boxes.
            text_threshold (float): The threshold for the text.
            nms_threshold (float): The threshold for NMS.
            verbose (bool): Whether to print verbose information.
        
        Returns:
            dict: A dictionary containing the following keys:
                - 'detections': The detected bounding boxes and masks.
                - 'bbox_image': The annotated image with bounding boxes.
                - 'mask_image': The annotated image with masks.
                - 'pcd_lst': A list of point clouds for each detected object.
        """
        img_points = points.reshape(colors.shape)
        bgr_colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)

        # Compute mask of points within depth range
        valid_mask = (img_points[:, :, 2] > depth_min) & (img_points[:, :, 2] < depth_max)
        if verbose:
            cv2.imshow('mask', valid_mask.astype(np.uint8) * 255)
            cv2.waitKey(0)

        bgr_colors[~valid_mask] = 0
        out_dict = self.segment(bgr_colors, classes, box_threshold, text_threshold, nms_threshold, verbose=True)

        detections = out_dict['detections']
        out_dict['pcd_lst'] = []
        for seg_id in range(len(detections)):
            pcd = o3d.geometry.PointCloud()
            mask = detections.mask[seg_id, :, :]

            if verbose:
                cv2.imshow('mask', mask.astype(np.uint8) * 255)
                cv2.waitKey(0)

            mask = mask & valid_mask

            mask_points = img_points[mask, :]
            mask_colors = colors[mask, :]

            pcd.points = o3d.utility.Vector3dVector(mask_points.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector(mask_colors.reshape(-1, 3) / 255.0)

            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
            pcd = pcd.select_by_index(ind)
            out_dict['pcd_lst'].append(pcd)

        return out_dict


if __name__ == '__main__':
    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "data/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "data/groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "data/sam_vit_h_4b8939.pth"

    # Predict classes and hyper-param for GroundingDINO
    SOURCE_IMAGE_PATH = "out_data/test_images/color_image.png"
    CLASSES = ["green leaf"]
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    ground_sam = GroundedSAM()

    out_dict = ground_sam.segment(image, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD)

    # save the annotated grounding dino image
    cv2.imwrite("groundingdino_bbox_image.jpg", out_dict['bbox_image'])

    # save the annotated sam image
    cv2.imwrite("sam_annotated_image.jpg", out_dict['mask_image'])
