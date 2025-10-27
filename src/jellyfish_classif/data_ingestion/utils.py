from pathlib import Path
import requests


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
