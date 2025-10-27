from typing import Dict, Any
from pathlib import Path

from jellyfish_classif.data_ingestion.utils import download_image
from jellyfish_classif.data_ingestion.utils import check_image


def process_observation(
    obs: Dict[str, Any],
    species_dir: Path,
    downloaded: int,
    max_images: int,
    image_size: str,
) -> int:
    """Processes a single observation, downloads its images.

    Args:
        obs (Dict[str, Any]): Observation dictionary from iNaturalist API
        species_dir (Path): Directory to save images for the species
        downloaded (int): Current count of downloaded images
        max_images (int): Maximum number of images to download for the species
        image_size (str): Requested image size (e.g. "medium", "large")

    Returns:
        int: Updated count of downloaded images
    """
    obs_id = obs["id"]
    photos = obs.get("photos", [])

    for i, photo in enumerate(photos):
        img_url = photo.get("url")
        if not img_url:
            continue

        # Modify URL to get the desired image size
        img_url = img_url.replace("square", image_size)

        filename = f"{obs_id}_{i}.jpg"
        filepath = species_dir / filename

        if filepath.exists():
            continue  # Skip if already downloaded

        if download_image(img_url, filepath):
            # Check if the image is valid
            if not check_image(filepath):
                try:
                    filepath.unlink()
                    print(f"Removed invalid image: {filepath.name}")
                except Exception as e:
                    print(f"Failed to remove invalid image {filepath}: {e}")
                continue

            downloaded += 1
            if downloaded >= max_images:
                break
    return downloaded
