from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

from jellyfish_classif.config.schema import DatasetConfig
from jellyfish_classif.config import Config

dataset_config = Config().dataset


class JellyfishDataset:
    """Manages loading and preprocessing of jellyfish image data."""

    def __init__(
        self,
        root_dir: str,
        config: DatasetConfig = dataset_config,
    ):
        self.root_dir = Path(root_dir)
        self.config = config
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self._label_map = None
        self._class_names: Optional[List[str]] = None
        self._num_classes: Optional[int] = None
        self.augment = self._build_augmentation_pipeline()

        self._load_from_directory()

    # ============================================================
    # === INITIALISATION =========================================
    # ============================================================

    def _load_from_directory(self) -> None:
        """Scan directory structure and load image paths and labels."""

        assert self.root_dir.exists(), f"Dataset directory not found: {self.root_dir}"

        # Each subfolder = one class
        self._class_names = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self._label_map = {name: idx for idx, name in enumerate(self._class_names)}
        self._num_classes = len(self._class_names)

        # Collect all image paths + labels
        for class_name in self._class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.image_paths.append(img_path.resolve())
                    self.labels.append(self._label_map[class_name])

        assert (
            len(self.image_paths) > 0
        ), f"No images found in dataset directory {self.root_dir}"

    # ============================================================
    # === AUGMENTATION ===========================================
    # ============================================================

    def _build_augmentation_pipeline(self) -> tf.keras.Sequential:
        """Builds image augmentation pipeline using Keras layers.

        Returns:
            tf.keras.Sequential: Augmentation pipeline
        """
        augmentation_layers = [
            layers.RandomFlip(mode="horizontal_and_vertical"),
            layers.RandomRotation(factor=0.10, fill_mode="reflect"),  # approx 18°
            layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
            layers.RandomContrast(factor=0.10),
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.15)),
            layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0)),
        ]
        return tf.keras.Sequential(augmentation_layers, name="jellyfish_augment")

    # ============================================================
    # === PREPROCESSING ==========================================
    # ============================================================

    def _preprocess_image(
        self, image_path: Path, label: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Load and preprocess one image: decode, resize, normalize

        Args:
            image_path (Path): Path to image file
            label (int): Corresponding label index

        Returns:
            Tuple[tf.Tensor, tf.Tensor] : (image tensor, label tensor)
        """

        def load_and_process(path):

            path = path.numpy().decode()

            img_bytes = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img_bytes, channels=3)

            img = tf.image.resize(img, [self.config.img_size, self.config.img_size])
            img = tf.clip_by_value(img / 255.0, 0.0, 1.0)
            return img

        image = tf.py_function(load_and_process, [image_path], tf.float32)
        image.set_shape([self.config.img_size, self.config.img_size, 3])
        return image, label

    def _augment_image(self, image: tf.Tensor, label: tf.Tensor):
        return self.augment(image, training=True), label

    # ============================================================
    # === DATASET CREATION =======================================
    # ============================================================

    def to_tf_dataset(
        self,
        image_paths: List[Path],
        labels: List[int],
        augment: bool = False,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """Create tf.data.Dataset from image paths and labels with options to shuffle, batch, augment

        Args:
            image_paths (list[Path]): Paths to image files
            labels (list[int]): Corresponding label indices
            augment (bool, optional): If True, proceeds augmentation. Defaults to False.
            shuffle (bool, optional): If True, shuffles images. Defaults to True.

        Returns:
            tf.data.Dataset : TensorFlow dataset ready for training/eval
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor([str(p) for p in image_paths]),
                tf.convert_to_tensor(labels),
            )
        )

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(image_paths), seed=self.config.seed
            )

        dataset = dataset.map(
            self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
        )

        if augment:
            dataset = dataset.map(
                self._augment_image, num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    # ============================================================
    # === SPLITTING ==============================================
    # ============================================================

    def split(
        self,
        train_ratio: float = 0.75,
        val_ratio: float = 0.15,
        test_ratio: float = 0.1,
    ) -> Tuple[List[Path], List[Path], List[Path], List[int], List[int], List[int]]:
        """Split the dataset into train/val/test sets only."""
        assert np.isclose(
            train_ratio + val_ratio + test_ratio, 1.0
        ), "Ratios must sum to 1"

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.image_paths,
            self.labels,
            test_size=(1 - train_ratio),
            stratify=self.labels,
            random_state=self.config.seed,
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_size),
            stratify=y_temp,
            random_state=self.config.seed,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ============================================================
    # === PROPERTIES =============================================
    # ============================================================

    @property
    def class_names(self) -> List[str]:
        """Class names in index order."""
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return self._num_classes

    @property
    def label_map(self) -> Dict[str, int]:
        """Mapping of class_name -> index."""
        return self._label_map
