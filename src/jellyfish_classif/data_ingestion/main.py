from typing import Dict, Any
from pathlib import Path
import csv
from time import sleep
from tqdm import tqdm


from jellyfish_classif.data_ingestion import observation, api, utils
from config import Config

download_config = Config().download


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

    metadata_dir = Path("data/metadata")
    images_dir = Path("data/images")

    species_list = download_config.species or []
    if not species_list:
        print("No taxon defined in config (config.yaml > download.species).")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metadata_dir / "jellyfish_dataset.csv"
    writer, csvfile = utils.create_csv_writer(csv_path)

    for species in tqdm(species_list, desc="Species"):
        download_species_images(
            species, writer
        )  # TODO: Ok, mais demander au chat comment faire pour éviter de gérer un contexte à l'intérieur d'un utilitaire

    csvfile.close()  # TODO: ajouter que si jamais existe déjà on refait pas
