import json
import os
import requests

from tqdm import tqdm


def download_file(url, output_path):
    """
    Downloads a file from the specified URL and saves it to the given output path.

    Args:
        url (str): The URL of the file to download.
        output_path (str): The local path where the file will be saved.

    Returns:
        None

    Example:
        download_file("https://example.com/file.zip", "file.zip")
    """

    if os.path.exists(output_path):
        print(f"File [{output_path}] already exists")
        return None

    print(f"Downlading from {url}....")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check if the request was successful

    # Get the total file size in bytes
    total_size = int(response.headers.get('content-length', 0))

    # Download the file with progress bar
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            progress_bar.update(len(data))

def main():
    HOME = os.getcwd()
    dino_chkpt = f"{HOME}/weights/groundingdino_swint_ogc.pth"
    dino_config = os.path.join(HOME, "configs/GroundingDINO_SwinT_OGC.py")
    sam_config = "vit_h"
    sam_chkpt = f"{HOME}/weights/sam_vit_h_4b8939.pth"

    download_data = {
        dino_chkpt: "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        sam_chkpt: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        dino_config: "https://raw.githubusercontent.com/chrizandr/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    }

    for path, url in download_data.items():
        download_file(url, path)

    seg_config = {
        "dino_chkpt": dino_chkpt,
        "dino_config": dino_config,
        "sam_config": sam_config,
        "sam_chkpt": sam_chkpt,
    }
    inpaint_config = {
        "inpaint_model": "stabilityai/stable-diffusion-2-inpainting"
    }
    with open(f"{HOME}/configs/segmentation_config.json", "w") as f:
        json.dump(seg_config, f, indent=4)

    with open(f"{HOME}/configs/inpainting_config.json", "w") as f:
        json.dump(inpaint_config, f, indent=4)


if __name__ == "__main__":
    main()