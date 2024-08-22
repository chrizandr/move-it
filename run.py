import os
import json
import argparse

from move_it.pipelines import SegmentationPipeline, InpaintingPipeline


HOME = os.getcwd()
try:
    segmentation_config = json.load(
        open(f"{HOME}/configs/segmentation_config.json"))
    inpainting_config = json.load(
        open(f"{HOME}/configs/inpainting_config.json"))
except FileNotFoundError:
    print("Cannot find config file, please run configure.py")


def main():
    parser = argparse.ArgumentParser(description="Move a specific object in a given image and generate a new image.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--class', dest="text", type=str, required=True, help="Class name to process.")
    parser.add_argument('--output', type=str, help="Path to save the output image.")
    parser.add_argument('--x', type=int, default=0, help="X coordinate offset for the object [can be negative].")
    parser.add_argument('--y', type=int, default=0, help="Y coordinate offset for the object [can be negative].")

    args = parser.parse_args()

    # Example usage of the parsed arguments
    print(f"Processing image: {args.image}")
    if args.output:
        print(f"Output will be saved to: {args.output}")
    if args.x == 0 and args.y == 0:
        print(f"Performing segmentation for prompt {args.text}")
        pipeline = SegmentationPipeline(segmentation_config)
        pipeline.run(args.image, args.output)
        print(f"Output segmentated image saved to: {args.output}")
    else:
        print(f"Moving the object {args.text} by x:{args.x} and y:{args.y}")
        pipeline = InpaintingPipeline(segmentation_config, inpainting_config)
        pipeline.run(args.image, args.output)
        print(f"Modified image saved to: {args.output}")


if __name__ == "__main__":
    main()