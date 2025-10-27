from typing import List, Dict, Any, Tuple
from pathlib import Path
import requests
import csv
import json


def load_species_list(path: Path) -> List[Dict[str, Any]]:
    """Loads the species list from a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        List[Dict[str, Any]]: Dictionary containing species information
        (name, preferred_common_name, taxon_id).
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_csv_writer(csv_path: Path) -> Tuple[csv.writer, Any]:
    """Creates a CSV writer and writes the header row.

    Args:
        csv_path (Path): Path to the CSV file to be written.

    Returns:
        Tuple[csv.writer, file]: The CSV writer object and the opened file handle.
    """

    # TODO: avec un with open ça ferme le fichier automatiquement
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csvfile = csv_path.open("w", newline="", encoding="utf-8")

    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "image_path",
            "taxon_id",
            "species_name",
        ]
    )
    return writer, csvfile  # return both the writer AND the opened file


def download_image(url: str, filepath: Path) -> bool:
    """
    Downloads an image from a URL and saves it to the specified filepath.

    Args:
        url (str): The URL of the image.
        filepath (Path): The local path to save the image.

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        filepath.write_bytes(response.content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False
