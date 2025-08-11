"""
Contains function to download data
"""

import os
import requests
import zipfile
from pathlib import Path


def get_data(
    data_dir_str: str = "data/",
    image_path_str: str = "pizza_steak_sushi",
    data_url_str: str = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    file_name_str: str = "pizza_steak_sushi.zip"
) -> None:
    """Downloads data from GitHub.

    Args:
        data_dir_str (str, optional): Path do data directory.
            Defaults to "../data/".
        image_path_str (str, optional): Name of the folder where data will
            be stored. Defaults to "pizza_steak_sushi".
        data_url_str (_type_, optional): Link to site from where data will
            be downloaded. Defaults to "https://github.com/mrdbourke/ \
            pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip".
        file_name_str (str, optional): Name of the downloaded file.
            Defaults to "pizza_steak_sushi.zip".
    """

    # Setup path to data folder
    data_dir = Path(data_dir_str)
    image_path = data_dir / image_path_str

    # Check if data folder exists
    if image_path.exists():
        print(f"{image_path} exists...")
    else:
        print(f"{image_path} does not exists, creating...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Check if data is already downloaded
    if len(list(image_path.glob("*/*/*"))) == 0:

        # Download data
        with open(data_dir / file_name_str, "wb") as f:
            print(f"Downloading {file_name_str}...")
            request = requests.get(data_url_str)
            f.write(request.content)

        # Unzip data
        with zipfile.ZipFile(data_dir / file_name_str, "r") as z:
            print(f"Extracting {file_name_str}...")
            z.extractall(image_path)

        # Remove zip file
        print(f"Deleting {file_name_str}...")
        os.remove(data_dir / file_name_str)
    else:
        print(f"Data in {image_path} already exits, skipping downloading and unzipping...")

    print("Finished getting data...")


if __name__ == "__main__":
    get_data()
