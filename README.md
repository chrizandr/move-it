# Move It!
A pipeline for detecting, moving, and inpainting objects in an image using segmentation and inpainting models.

The models used in this project include:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - For generating bounding boxes using text prompts
- [SAM](https://github.com/facebookresearch/segment-anything) - For generating sementation masks within a bounding box
- [Stable-Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) - For inpainting the gap left behind after object is moved.

NOTE: Project has only been tested for **Python3.10.x**. Please use compatible version to avoid dependency issues.

## Setup
Clone the repo and install the dependencies
```
git clone https://github.com/chrizandr/move_it.git
cd /content/move_it
pip install -q -r requirements.txt
```
NOTE: The models don't need to be installed separately as they are forked and added to the requirements file.

### Run `configure.py` to download the relevant model checkpoints and create configs.
```
python configure.py
```
If you want to use alternate checkpoints for the models, you can edit the config files in `configs/` accordingly. I have provided sample configs. If you plan to use the defualt settings, no need to edit anything after running `configure`.

## Inference

There are two tasks possible, each with its own pipeline.

The first task is unconstrained segmentation. This task identifies an object in the image based on a text prompt and segments it. A segmentation color(red) mask applied on the object.
```
python run.py --image ./example.jpg \
              --class shelf \
              --output ./generated.png
```
The second task is to move the object given in the text prompt by an x and y offset. This task calls the segmentation pipeline first, then moves the object and inpaints the gap left by the object in its former position.
```
python run.py --image ./example.jpg \
              --class shelf \
              --x 80 —-y 0 \
              --output ./generated.png
```

## Results

Results for the sample images along with a other experiments are present in the [Demo notebook](https://github.com/chrizandr/move_it/blob/main/Demo.ipynb). This also gives relevant code to run the pipelines in python code rather than from the command line.
