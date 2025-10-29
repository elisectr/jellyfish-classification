import tensorflow as tf
from typing import Dict, Callable

from jellyfish_classif.config.schema import ModelConfig
from jellyfish_classif.config import Config

model_config = Config().model


class ModelFactory:
    """Factory to create transfer learning models."""

    # ===============================================================
    # === AVAILABLE ARCHITECTURES ===================================
    # ===============================================================

    # Available architectures
    _MODELS: Dict[str, Callable] = {
        "ResNet50": tf.keras.applications.ResNet50,
        "ResNet101": tf.keras.applications.ResNet101,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "VGG16": tf.keras.applications.VGG16,
        "InceptionV3": tf.keras.applications.InceptionV3,
    }

    # Corresponding preprocessing functions
    _PREPROCESSORS: Dict[str, Callable] = {
        "ResNet50": tf.keras.applications.resnet.preprocess_input,
        "ResNet101": tf.keras.applications.resnet.preprocess_input,
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "VGG16": tf.keras.applications.vgg16.preprocess_input,
        "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
    }

    def __init__(self, config: ModelConfig = model_config):
        self.config = config

    # ===============================================================
    # === PUBLIC ====================================================
    # ===============================================================

    def create_model(self, model_name: str) -> tf.keras.Model:
        """Create and return a Keras model based on the specified architecture :
        non trainable base model + classification head.

        Args:
            model_name: Architecture name (e.g., 'ResNet50'). To see available models,
                        use get_available_models().

        Returns:
            Keras model (not compiled).
        """
        if model_name not in self._MODELS:
            raise ValueError(
                f"Model '{model_name}' unknown. "
                f"Availables: {self.get_available_models()}"
            )

        # Create base model
        base_model = self._load_backbone(model_name)
        base_model.trainable = False

        # Build full model
        model = self._assemble_model(model_name, base_model)

        return model

    def compile_model(
        self,
        model: tf.keras.Model,
        learning_rate: float | None = None,
    ) -> tf.keras.Model:
        """Compile the Keras model with Adam optimizer, loss, and metrics."""

        lr = learning_rate or self.config.learning_rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=self.config.loss_function,
            metrics=self.config.metrics,
        )
        return model

    def unfreeze_top_layers(self, model: tf.keras.Model):
        """Unfreeze the top N layers of the base model for fine-tuning."""
        base_model = model.get_layer(index=1)  # after Input/Preprocessing
        base_model.trainable = True

        total_layers = len(base_model.layers)
        freeze_until = max(0, total_layers - self.config.fine_tune_layers)

        for i, layer in enumerate(base_model.layers):
            layer.trainable = i >= freeze_until

    # ===============================================================
    # === INTERNAL BUILDING BLOCKS ==================================
    # ===============================================================

    def _load_backbone(
        self,
        model_name: str,
    ) -> tf.keras.Model:
        """Create pre-trained base model."""
        model_class = self._MODELS[model_name]

        base_model = model_class(
            include_top=False,
            weights="imagenet",
            input_shape=self.config.input_shape,
        )

        return base_model

    def _add_classification_head(self, x: tf.Tensor) -> tf.Tensor:
        """Add the top layers for classification (head)."""
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.config.num_classes, activation="softmax")(
            x
        )
        return outputs

    def _assemble_model(
        self, model_name: str, base_model: tf.keras.Model
    ) -> tf.keras.Model:
        """Build the complete model: preprocessing + base + head."""
        inputs = tf.keras.Input(shape=self.config.input_shape)

        # Preprocessing (architecture specific)
        preprocess_fn = self._PREPROCESSORS[model_name]
        x = preprocess_fn(inputs)

        # Base model
        x = base_model(x)

        # Classification head
        outputs = self._add_classification_head(x)

        # Final assembled model
        model = tf.keras.Model(inputs, outputs, name=model_name)
        return model

    # ===============================================================
    # === PROPERTIES ================================================
    # ===============================================================

    @property
    def is_compiled(self) -> bool:
        """Check if the model is compiled."""
        return hasattr(self, "optimizer")

    @property
    def available_models(self) -> list[str]:
        """Get list of available model names."""
        return list(self._MODELS.keys())
