from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path


class JellyfishDataset:
    """Manages loading and preprocessing of jellyfish image data."""

    def __init__(
        self,
        csv_path: str,
        config,
    ):
        self.csv_path = Path(csv_path)
        self.config = config
        self.image_paths = None
        self.labels = None
        self.label_map = None
        self.class_names: Optional[List[str]] = None
        self.num_classes: Optional[int] = None

        self._load_metadata()
        self.augment = tf.keras.Sequential(
            [
                layers.RandomFlip(mode="horizontal_and_vertical"),
                layers.RandomRotation(factor=0.10, fill_mode="reflect"),  # approx 18°
                layers.RandomZoom(height_factor=0.10, width_factor=0.10),
                layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
                layers.RandomContrast(factor=0.10),
                layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.15)),
                layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0)),
            ],
            name="jellyfish_augment",
        )

    def _load_metadata(self):
        """Loads image paths and labels from CSV and creates label->index mapping"""

        df = pd.read_csv(self.csv_path)

        # Ensure required columns exist
        assert (
            "image_path" in df.columns and "species_name" in df.columns
        ), "CSV must contain 'image_path' and 'species_name'"

        self.image_paths = [Path(p) for p in df["image_path"].values.tolist()]
        labels_raw = df["species_name"].values

        # Create mapping label->index
        self.class_names = sorted(set(labels_raw))
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.labels = [self.label_map[label] for label in labels_raw]

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
        image = tf.io.read_file(str(image_path))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.config.img_size, self.config.img_size])
        image = image / 255.0  # Normalisation [0,1]

        return image, label

    # TODO later : add augmentations
    def _augment_image(self, image: tf.Tensor, label: tf.Tensor):
        image = self.augment(image, training=True)
        return image, label

    def compute_class_weights(self) -> Dict[int, float]:
        """Compute class weights to handle class imbalance.

        Returns:
            Dict[int, float]: Mapping class_index -> weight
        """
        classes = np.unique(self.labels)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.labels,
        )
        class_weights = {int(cls): float(w) for cls, w in zip(classes, weights)}
        return class_weights  # TODO: objet tf qui fait ça ?

    def _prepare_dataset(
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
            ([str(p) for p in image_paths], labels)
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

    # TODO: shuffle, Augment ...

    def split_and_prepare(
        self,
        train_ratio: float = 0.75,
        val_ratio: float = 0.15,
        test_ratio: float = 0.1,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Splits dataset into train/val/test and prepares tf.data.Dataset for each


        Args:
            train_ratio (float, optional): Train split ratio. Defaults to 0.8.
            val_ratio (float, optional): Validation split ratio. Defaults to 0.1.
            test_ratio (float, optional): Test split ratio. Defaults to 0.1.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
        """

        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

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

        train_ds = self._prepare_dataset(X_train, y_train, augment=True, shuffle=True)
        val_ds = self._prepare_dataset(X_val, y_val, augment=False, shuffle=False)
        test_ds = self._prepare_dataset(X_test, y_test, augment=False, shuffle=False)

        return train_ds, val_ds, test_ds

    # TODO: séparer split et prepare

    # TODO: @property
    def get_class_names(self) -> List[str]:
        """Returns class names in index order."""
        return self.class_names

    def get_num_classes(self) -> int:
        """Returns number of classes."""
        return self.num_classes

    def get_label_map(self) -> Dict[str, int]:
        """Returns mapping of class_name -> index."""
        return self.label_map
