import tensorflow as tf
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from jellyfish_classif.config.schema import TrainingConfig
from jellyfish_classif.config import Config

training_config = Config().training


class Trainer:
    """Handles model training, logging, and saving."""

    def __init__(
        self,
        model: tf.keras.Model,
        config: TrainingConfig = training_config,
    ):
        """
        Args:
            model: Compiled Keras model
            config: Training config
        """
        self.model = model
        self.config = config
        self.experiment_name = self._generate_experiment_name()

        # Create output directory
        self.output_dir = Path("outputs") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Training history
        self.history = None

        logging.info(f"Trainer initialized for: {self.experiment_name}")
        logging.info(f"Output dir: {self.output_dir}")

    # ==============================================================
    # === CORE =====================================================
    # ==============================================================

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        class_weight: Optional[Dict[int, float]] = None,
        epochs: int = 10,
        initial_epoch: int = 0,
    ) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            class_weight: Optional class weights for imbalance correction
            initial_epoch: Initial epoch (if previous training)

        Returns:
            Keras training history
        """
        logging.info("=" * 50)
        logging.info("STARTING TRAINING")
        logging.info("=" * 50)

        # Save config
        self._save_metadata()

        # Prepare callbacks
        callbacks = self._setup_callbacks()

        # Training
        try:
            logging.info(f"Training on {epochs} epochs...")

            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                verbose=self.config.verbose,
                class_weight=class_weight,
            )

            logging.info("Training done !")

        except Exception as e:
            logging.error(f"Error while training : {e}")
            raise

        # Post-processing
        self._post_training()

        return self.history

    # ==============================================================
    # === LOGGING & METADATA =======================================
    # ==============================================================

    def _generate_experiment_name(self) -> str:
        """Generate a unique experiment name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.name
        return f"{model_name}_{timestamp}"

    def _setup_logging(self, log_filename: str = "training.log"):
        """Setup logging to file and console."""

        log_file = self.output_dir / log_filename
        logger = logging.getLogger()

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Global logging level
        logger.setLevel(logging.INFO)

        # Console handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    def _save_metadata(self):
        """Save experiment metadata."""
        metadata = {
            "experiment_name": self.experiment_name,
            "model_name": self.model.name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "epochs": self.config.epochs,
                "early_stopping_patience": self.config.early_stopping_patience,
            },
            "model_config": {
                "total_params": int(self.model.count_params()),
                "trainable_params": int(
                    sum(
                        tf.keras.backend.count_params(w)
                        for w in self.model.trainable_weights
                    )
                ),
            },
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Saved metadata to : {metadata_path}")

    # ==============================================================
    # === CALLBACKS, POST PROC =====================================
    # ==============================================================

    def _setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Setup training callbacks."""
        callbacks = []

        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.config.checkpoint_monitor,
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=self.config.verbose,
        )
        callbacks.append(early_stopping)
        logging.info(
            f"EarlyStopping: monitor={self.config.checkpoint_monitor}, "
            f"patience={self.config.early_stopping_patience}"
        )

        # Model Checkpoint
        checkpoint_path = self.output_dir / "best_model.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=self.config.checkpoint_monitor,
            save_best_only=True,
            mode=self.config.checkpoint_mode,
            verbose=self.config.verbose,
        )
        callbacks.append(checkpoint)
        logging.info(f"ModelCheckpoint: {checkpoint_path}")

        # CSV Logger
        csv_path = self.output_dir / "training_history.csv"
        csv_logger = tf.keras.callbacks.CSVLogger(str(csv_path))
        callbacks.append(csv_logger)

        # Reduce Learning Rate on Plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, verbose=self.config.verbose
        )
        callbacks.append(reduce_lr)

        return callbacks

    def _post_training(self):  # TODO: laisser user choisir
        """After training: save model, plots, metrics."""
        # Save final model
        final_model_path = self.output_dir / "final_model.keras"
        self.model.save(final_model_path)
        logging.info(f"Final model saved: {final_model_path}")

        # Plot training curves
        self._plot_training_curves()

        # Log final metrics
        self._log_final_metrics()

    # ==============================================================
    # === VISUALS & METRICS ========================================
    # ==============================================================

    def _plot_training_curves(self):
        """Plot and save training curves."""
        if self.history is None:
            return

        history = self.history.history

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Training History - {self.experiment_name}", fontsize=16)

        # Loss
        axes[0].plot(history["loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(history["accuracy"], label="Train Accuracy", linewidth=2)
        axes[1].plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save
        curves_path = self.output_dir / "training_curves.png"
        plt.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close()

        logging.info(f"Training curves saved: {curves_path}")

    def _log_final_metrics(self):
        """Log final training metrics to logging file."""
        if self.history is None:
            return

        history = self.history.history

        # Best epoch based on val_loss
        best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1

        logging.info("=" * 50)
        logging.info("FINAL RESULTS")
        logging.info("=" * 50)
        logging.info(f"Best epoch: {best_epoch}/{history.epoch[-1]}")
        logging.info(f"Train Loss: {history['loss'][best_epoch-1]:.4f}")
        logging.info(f"Val Loss: {history['val_loss'][best_epoch-1]:.4f}")
        logging.info(f"Train Accuracy: {history['accuracy'][best_epoch-1]:.4f}")
        logging.info(f"Val Accuracy: {history['val_accuracy'][best_epoch-1]:.4f}")

        # Overfitting check
        overfit_gap = (
            history["accuracy"][best_epoch - 1]
            - history["val_accuracy"][best_epoch - 1]
        )
        if overfit_gap > 0.1:
            logging.warning(f"Possible overfitting detected (gap: {overfit_gap:.2%})")

        logging.info("=" * 50)
