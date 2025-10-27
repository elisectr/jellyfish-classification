from typing import Dict, Any
import os
import csv
from time import sleep
from tqdm import tqdm


from jellyfish_classif.data_ingestion import observation, api, utils
from config import DownloadConfig  # TODO: gérer les imports

download_config = DownloadConfig()


def download_species_images(species: Dict[str, Any], writer: csv.writer) -> None:
    """Downloads images for a given species and writes metadata to CSV.

    Args:
        species (Dict[str, Any]): Species information dictionary
        writer (csv.writer): CSV writer object
    """
    sci_name = species["name"]
    taxon_id = species["taxon_id"]
    common_name = species.get("preferred_common_name", "")

    print(f"\n🔍 {sci_name} ({common_name})")

    species_dir = os.path.join("data/images", sci_name.replace(" ", "_"))
    os.makedirs(species_dir, exist_ok=True)

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
                    writer,
                    species,
                    downloaded,
                    download_config.max_images_per_species,
                    download_config.image_size,
                )
                pbar.update(1)
                if downloaded >= download_config.max_images_per_species:
                    break

            page += 1
            sleep(download_config.api_sleep_time)

    print(f"{downloaded} images téléchargées pour {sci_name}")


def download_all_species() -> None:
    """Downloads images for all species listed in the species JSON file and writes CSV metadata file."""
    species_list = utils.load_species_list("data/metadata/species.json")
    os.makedirs("data/images", exist_ok=True)
    writer, csvfile = utils.create_csv_writer("data/metadata/jellyfish_dataset.csv")

    for species in tqdm(species_list, desc="Espèces"):
        download_species_images(
            species, writer
        )  # TODO: Ok, mais demander au chat comment faire pour éviter de gérer un contexte à l'intérieur d'un utilitaire

    csvfile.close()  # TODO: ajouter que si jamais existe déjà on refait pas
