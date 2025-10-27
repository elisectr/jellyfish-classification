from pathlib import Path
import requests
import tensorflow as tf

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



def check_image(filepath: Path) -> bool:
    """
    Checks if an image file is valid by attempting to decode it with TensorFlow.

    Args:
        filepath (Path): Path to the image file.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        img_data = tf.io.read_file(str(filepath))
        _ = tf.image.decode_jpeg(img_data, channels=3)
        return True
    except tf.errors.InvalidArgumentError as e:
        print(f"Invalid image (decode error): {filepath} ({e})")
        return False
    except Exception as e:
        print(f"Error on {filepath}: {e}")
        return False