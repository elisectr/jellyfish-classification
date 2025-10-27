from typing import Dict, Any
from pathlib import Path
from time import sleep
from tqdm import tqdm


from jellyfish_classif.data_ingestion import observation, api
from config import Config

download_config = Config().download


def download_species_images(species: Dict[str, Any]) -> None:
    """Downloads images for a given species.

    Args:
        species (Dict[str, Any]): Species information dictionary
    """
    sci_name = species["name"]
    taxon_id = species["taxon_id"]
    common_name = species.get("preferred_common_name", "")

    print(f"\n🔍 {sci_name} ({common_name})")

    species_dir = Path("data/images") / sci_name.replace(" ", "_")
    species_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    page = 1

    with tqdm(
        total=download_config.max_images_per_species, desc=f"{sci_name}", leave=False
    ) as pbar:
        while downloaded < download_config.max_images_per_species:
            observations = api.fetch_observations(
                taxon_id,
                page,
                download_config.per_page,
            )
            if not observations:
                break

            for obs in observations:
                downloaded = observation.process_observation(
                    obs,
                    species_dir,
                    downloaded,
                    download_config.max_images_per_species,
                    download_config.image_size,
                )
                pbar.update(1)
                if downloaded >= download_config.max_images_per_species:
                    break

            page += 1
            sleep(download_config.api_sleep_time)

    print(f"{downloaded} images downloaded for {sci_name}")


def download_all_species() -> None:
    """Downloads images for all species listed in the config file."""

    images_dir = Path("data/images")

    species_list = download_config.species or []
    if not species_list:
        print("No taxon defined in config (config.yaml > download.species).")
        return

    images_dir.mkdir(parents=True, exist_ok=True)

    for species in tqdm(species_list, desc="Species"):
        sci_name = species["name"]
        species_dir = images_dir / sci_name.replace(" ", "_")
        if species_dir.exists() and any(species_dir.glob("*.jpg")):
            print(f"Dataset already exists for {sci_name}, skipping download.")
            continue

        download_species_images(species)
