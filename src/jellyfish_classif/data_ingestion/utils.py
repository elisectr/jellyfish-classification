from typing import List, Dict, Any
import requests
import os
import csv
import json


def load_species_list(path: str) -> List[Dict[str, Any]]:
    """Loads the species list from a JSON file.

    Args:
        path (str): path to the JSON file.

    Returns:
        List[Dict[str, Any]]: Dictionary containing species information
        (name, preferred_common_name and taxon_id).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_csv_writer(csv_path: str) -> csv.writer:
    """Creates a CSV writer and writes the header row.

    Args:
        csv_path (str): Path to the CSV file to be written.

    Returns:
        csv.writer: CSV writer object.
    """
    # TODO: avec un with open ça ferme le fichier automatiquement
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csvfile = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "image_path",
            "taxon_id",
            "species_name",
        ]
    )
    return writer, csvfile  # return both the writer AND the opened file


def download_image(url: str, filepath: str) -> bool:
    """
    Downloads an image from a URL and saves it to the specified filepath.

    Args:
        url (str): The URL of the image.
        filepath (str): The local path to save the image.

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
        return True
    return False
