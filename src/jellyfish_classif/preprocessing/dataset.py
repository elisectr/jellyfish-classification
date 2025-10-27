from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path


class JellyfishDataset:
    """Manages loading and preprocessing of jellyfish image data."""

    def __init__(
        self,
        root_dir: str,
        config,
    ):
        self.root_dir = Path(root_dir)
        self.config = config
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self._label_map = None
        self._class_names: Optional[List[str]] = None
        self._num_classes: Optional[int] = None

        self._load_from_directory()
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

    def _load_from_directory(self)-> None:
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
                    self.image_paths.append(img_path)
                    self.labels.append(self._label_map[class_name])

        assert (
            len(self.image_paths) > 0
        ), f"No images found in dataset directory {self.root_dir}"

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
        return class_weights

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

    # TODO: methode pour shuffle, Augment ...

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
