import os
import numpy as np
from PIL import Image

from move_it.segmentation import SegmentationModel
from move_it.inpainting import InpaintingModel


HOME = os.getcwd()


class SegmentationPipeline:
    def __init__(self, segmentation_config) -> None:
        self.model = SegmentationModel(**segmentation_config)

    def run(self, image_path, text, output_path):
        boxes, logits, phrases, image_np = self.model.detect(
            image_path, text)

        annotated_box = self.model.annotate_bbox(image_np, boxes, logits, phrases)
        box_pil = Image.fromarray(annotated_box)
        box_pil.save(f"{os.path.join(HOME, "logs/box_" + os.path.basename(image_path))}")

        mask = self.model.segment(image_np, boxes)[0][0]
        annotated_mask = self.model.annotate_mask(mask, image_np)
        mask_pil = Image.fromarray(annotated_mask)
        box_pil.save(output_path)

        return mask, image_np


class InpaintingPipeline:
    def __init__(self, segmentation_config, inpainting_config) -> None:
        self.seg_pipeline = SegmentationPipeline(**segmentation_config)
        self.model = InpaintingModel(**inpainting_config)

    def move_object(self, image_np, mask, x_off, y_off):
        h, w = image_np.shape

        cords = (mask > 0).nonzero()
        x, y = cords[0], cords[1]
        xmin, ymin = x.min(), y.min()
        xmax, ymax = x.max(), y.max()

        cropped_object = image_np[xmin:xmax+1, ymin:ymax+1]
        cropped_mask = mask[xmin:xmax+1, ymin:ymax+1][:, :, None].astype(np.uint8)

        # # Fix rotation and change coords accordingly
        # object_object = rotate(object_object, 10, resize=True, mode='constant', cval=0)
        # object_mask = rotate(object_mask, 10, resize=True, mode='constant', cval=0)

        # Offset cords but make sure it is not out of the image
        x1, x2 = np.clip(xmin+x_off, 0, h), np.clip(xmax+x_off, 0, h)
        y1, y2 = np.clip(ymin+y_off, 0, w), np.clip(ymax+y_off, 0, w)

        # print(x1_off, x2_off, y1_off, y2_off)
        new_image = image_np.copy()
        new_image_region = new_image[x1:x2+1, y1:y2+1]

        background_region = np.multiply(new_image_region, (1-cropped_mask))
        masked_object = np.multiply(cropped_object, cropped_mask)
        combined_region = background_region + masked_object

        new_image[x1:x2, y1:y2] = combined_region[0:x2-x1, 0:y2-y1]

        return new_image

    def run(self, image_path, text, offsets, output_path):
        mask_img_path = f"{os.path.join(HOME, "logs/mask_" + os.path.basename(image_path))}"
        x_off, y_off = offsets

        mask, image_np = self.seg_pipeline.run(image_path, text, mask_img_path)
        new_image = self.move_object(mask, image_np, x_off, y_off)

        inpainted_img = self.model.inpaint(new_image, mask, rectangle_mask=True)
        inpainted_img.save(output_path)
