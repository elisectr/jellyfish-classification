from typing import Dict, Any
import os
import csv

from jellyfish_classif.data_ingestion.utils import download_image


def process_observation(
    obs: Dict[str, Any],
    species_dir: str,
    writer: csv.writer,
    species_info: Dict[str, str],
    downloaded: int,
    max_images: int,
    image_size: str,
) -> int:
    """Processes a single observation, downloads its images, and writes metadata to CSV.

    Args:
        obs (Dict[str, Any]): Observation dictionary from iNaturalist API
        species_dir (str): Directory to save images for the species
        writer (csv.writer): CSV writer object
        species_info (Dict[str, str]): Species information dictionary
        downloaded (int): Current count of downloaded images
        max_images (int): Maximum number of images to download for the species

    Returns:
        int: Updated count of downloaded images
    """
    obs_id = obs["id"]
    photos = obs.get("photos", [])

    for i, photo in enumerate(photos):
        img_url = photo.get("url")
        if img_url:
            img_url = img_url.replace("square", image_size)
        else:
            continue

        filename = f"{obs_id}_{i}.jpg"
        filepath = os.path.join(species_dir, filename)

        if os.path.exists(filepath):
            continue  # Skip if already downloaded
        elif download_image(img_url, filepath):
            writer.writerow(
                [
                    filename,
                    species_info["taxon_id"],
                    species_info["name"],
                ]
            )
            downloaded += 1
            if downloaded >= max_images:
                break
    return downloaded


# TODO: enlever metadata
