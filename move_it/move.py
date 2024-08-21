from PIL import Image
from skimage.transform import rotate
import numpy as np


def move_object(image_np, mask, x_off, y_off):
    """
    Moves an object within an image to a new location specified by offset values.

    This function takes an image and a corresponding mask, identifies the object
    based on the mask, and moves it by the specified x and y offsets. The object
    is repositioned within the and cropped to remain within image boundaries.
    The function returns a new image with the object in its new location.

    Args:
        image_np(np.ndarray): The original image as a NumPy array.
        mask(np.ndarray): A binary mask indicating the object's location within the image.
        x_off(int): The horizontal offset by which to move the object.
        y_off(int): The vertical offset by which to move the object.

    Returns:
        np.ndarray: A new image with the object moved to the specified location.
    """
    cords = (mask > 0).nonzero()
    x_s, y_s = cords[0], cords[1]
    xmin, ymin = x_s.min(), y_s.min()
    xmax, ymax = x_s.max(), y_s.max()

    object_img = image_np[xmin: min(xmax+1, image_np.shape[0]), ymin: min(ymax+1, image_np.shape[1])]
    object_mask = mask[xmin: min(xmax+1, mask.shape[0]), ymin: min(ymax+1, mask.shape[1])]

    # # Fix rotation and change coords accordingly
    # object_img = rotate(object_img, 10, resize=True, mode='constant', cval=0)
    # object_mask = rotate(object_mask, 10, resize=True, mode='constant', cval=0)

    # Offset cords but make sure it is not out of the image
    x1_off, x2_off = max(0, xmin+x_off), min(xmax+1+x_off, image_np.shape[0])
    y1_off, y2_off = max(0, ymin+y_off), min(ymax+1+y_off, image_np.shape[1])
    print(x1_off, x2_off, y1_off, y2_off)
    new_image = image_np.copy()

    new_image_region = new_image[x1_off:x2_off, y1_off:y2_off].copy()
    object_mask_region = object_mask[0:x2_off-x1_off, 0:y2_off-y1_off]

    masked_region = np.multiply(
        new_image_region, (1-object_mask)[:, :, None].astype(np.uint8))
    masked_object = np.multiply(
        object_img, object_mask[:, :, None].astype(np.uint8))
    combined_region = masked_region + masked_object

    new_image[x1_off:x2_off, y1_off:y2_off] = combined_region[0:x2_off-x1_off, 0:y2_off-y1_off]

    return new_image
