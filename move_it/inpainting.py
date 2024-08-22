from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline


class InpaintingModel:
    def __init__(self, inpaint_model):
        """
        A class that utilizes the Stable Diffusion Inpainting model to perform image inpainting.

        The class provides functionality to mask regions of an image, either using a tight bounding
        rectangle around the masked area or using the original mask, and then fills in the masked
        region with background content using a text prompt.

        Args:
            inpaint_model (str): The name or path of the pre-trained inpainting model to be used.

        Methods:
            rectangle_mask_image(image_np, mask, pad=10):


            inpaint(image_np, mask, rectangle_mask=True):

        """
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpaint_model,
            torch_dtype=torch.float16,
        )
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")

    def rectangle_mask_image(self, image_np, mask, pad=10):
        """
        Creates a rectangular mask around the masked region and returns the masked image
        and the new rectangular mask.

        Args:
            image_np (np.ndarray): The original image as a NumPy array.
            mask (np.ndarray): A binary mask where the region to be inpainted is marked.
            pad (int, optional): Padding around the rectangular mask. Defaults to 10.

        Returns:
            masked_image (np.ndarray): The image with the mask applied.
            box_mask (np.ndarray): The new rectangular binary mask.
        """
        image_np.shape, mask.shape
        box_mask = np.zeros_like(mask)

        x_s, y_s = (mask > 0).nonzero()
        xmin, xmax = x_s.min(), x_s.max()
        ymin, ymax = y_s.min(), y_s.max()

        box_mask[max(0, xmin-pad):xmax+1+pad, max(0, ymin-pad):ymax+1+pad] = 1
        masked_image = image_np.copy()
        masked_image[(mask > 0).nonzero()] = 255

        return masked_image, box_mask

    def inpaint(self, image_np, mask, rectangle_mask=True, strength=1):
        """
        Performs inpainting on the image using the specified mask, filling in the masked
        area with content that matches the surrounding background.

        Args:
            image_np (np.ndarray): The original image as a NumPy array.
            mask (np.ndarray): A binary mask where the region to be inpainted is marked.
            rectangle_mask (bool, optional): If True, use a rectangular mask around the masked
            region. If False, use the original mask. Defaults to True.

        Returns:
            PIL.Image.Image: The inpainted image as a PIL Image.
        """
        prompt = "background"

        # image and mask_image should be PIL images.
        # The mask structure is white for inpainting and black for keeping as is
        if rectangle_mask:
            masked_image, box_mask = self.rectangle_mask_image(image_np, mask)
            mask_image = Image.fromarray(box_mask)
            image = Image.fromarray(masked_image)
        else:
            mask_image = Image.fromarray(mask)
            image = Image.fromarray(image_np)

        output = self.pipe(prompt=prompt, image=image,
            mask_image=mask_image,
            stength=strength, guidance_scale=10
        ).images[0]

        return output
