import os
import json

from move_it.segmentation import SegmentationModel
from move_it.inpainting import InpaintingModel

HOME = os.getcwd()
try:
    segmentation_config = json.read(open(f"{HOME}/configs/segmentation_config.json"))
    inpainting_config = json.read(open(f"{HOME}/configs/inpainting_config.json"))
except FileNotFoundError:
    print("Cannot find config file, please run configure.py")

class SegmentationPipeline:
    pass

class InpaintingPipeline:
    pass