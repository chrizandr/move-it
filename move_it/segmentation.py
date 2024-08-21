from PIL import Image
import torch
import numpy as np

from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops


class SegmentationModel:
    def __init__(self, dino_config, dino_chkpt, sam_config, sam_chkpt, device):
        """
        A class that combines GroundingDINO for object detection and SAM (Segment Anything Model) for segmentation, enabling detection, annotation, and segmentation of objects within an image based on a textual prompt.

        Args:
            dino_config (str): Path to the GroundingDINO model configuration file.
            dino_chkpt (str): Path to the GroundingDINO model checkpoint file.
            sam_config (str): The configuration key for the SAM model from the registry.
            sam_chkpt (str): Path to the SAM model checkpoint file.
            device (str): The device to run the models on (e.g., 'cpu', 'cuda').
        """
        self.device = device
        self.dino = load_model(dino_config, dino_chkpt, device)
        self.sam = SamPredictor(sam_model_registry[sam_config](
            checkpoint=sam_chkpt).to(device))

    def detect(self, image_path, text, box_threshold=0.3, text_threshold=0.25):
        """
        Detects objects in the image based on the provided textual prompt.

        Args:
            image_path(str): Path to the input image file.
            text(str): Text prompt describing the object to detect.
            box_threshold(float, optional): Confidence threshold for object detection boxes. Defaults to 0.3.
            text_threshold(float, optional): Confidence threshold for text detection. Defaults to 0.25.

        Returns:
            boxes(np.ndarray): Detected bounding boxes.
            logits(np.ndarray): Logits for the detected boxes.
            phrases(list): Phrases associated with the detected boxes.
            image_np(np.ndarray): The original image in NumPy array format.
        """
        # Dino seems to perform better when the input text is hyphenated instead of spaced.
        # Spaced text treats each word as a separate class and detects separate objects.
        image_np, image = load_image(image_path)
        text = "-".join(text.split())
        boxes, logits, phrases = predict(
            model=self.dino,
            image=image,
            caption=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        return boxes, logits, phrases, np.array(image_np)

    def annotate_bbox(self, image_np, boxes, logits, phrases):
        """
        Annotates an image with bounding boxes and corresponding text.

        Args:
            image_np (np.ndarray): The original image as a NumPy array.
            boxes (np.ndarray): Bounding boxes to annotate.
            logits (np.ndarray): Logits associated with the bounding boxes.
            phrases (list): Phrases corresponding to the detected objects.

        Returns:
            np.ndarray: The annotated image with bounding boxes and text.
        """
        annotated_frame = annotate(
            image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        return annotated_frame

    def segment(self, image_np, boxes):
        """
        Segments objects within the image based on detected bounding boxes.

        Args:
            image_np(np.ndarray): The original image as a NumPy array.
            boxes(np.ndarray): Bounding boxes used for segmentation.

        Returns:
            torch.Tensor: A tensor containing the segmentation masks.
        """
        self.sam.set_image(image_np)
        H, W, _ = image_np.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.sam.transform.apply_boxes_torch(
            boxes_xyxy.to(self.device), image_np.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.cpu()

    def annotate_mask(self, mask, image_np):
        """
        Overlays a segmentation mask onto the original image.

        Args:
            mask(torch.Tensor): The segmentation mask to overlay.
            image_np(np.ndarray): The original image as a NumPy array.

        Returns:
            np.ndarray: The image with the segmentation mask overlayed.
        """
        color = np.array([220/255, 20/255, 60/255, 0.6])  # Red mask
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        image_pil = Image.fromarray(image_np).convert("RGBA")
        mask_image_pil = Image.fromarray(
            (mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(image_pil, mask_image_pil))
