# The keyword classification model used to identify the potential keywords for non-categorised records.
import pandas as pd
import numpy as np
import tensorflow as tf

# this is an IDE issue reported 6 years ago and has not been fixed (https://youtrack.jetbrains.com/issue/PY-34174) and
# (https://youtrack.jetbrains.com/issue/PY-53599/tensorflow.keras-subpackages-are-unresolved-in-Tensorflow-2.6.0), so here I simply silence it locally with type:ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout  # type: ignore
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras import backend as backend  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

from typing import Dict, Callable, Any, Tuple, Optional, List
import os
from pathlib import Path

from data_discovery_ai.ml.preprocessor import KeywordPreprocessor
import mlflow  # type: ignore

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from data_discovery_ai.config.constants import KEYWORD_FOLDER
from data_discovery_ai import logger

mlflow.tensorflow.autolog()
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Keyword Classification Model")


def focal_loss(
    gamma: float, alpha: float
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Creates a focal loss function with specified gamma and alpha parameters. To address imbalanced class.
    Input:
        gamma: int. parameter that controls the down-weighting of easy examples. Higher gamma increases the effect.
        alpha: float. Should be in the range of [0,1]. Balancing factor for positive vs negative classes.
    Output:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: A focal loss function that takes in `y_true` (true labels) and `y_pred` (predicted labels) tensors and returns the focal loss as a tensor.
    """

    def focal_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        epsilon = backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = -alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)

    return focal_loss_fixed


def train_keyword_model(
    model_name: str, keywordPreprocessor: KeywordPreprocessor
) -> Tuple[Sequential, Any]:
    train_test_data = keywordPreprocessor.train_test_data
    trainer_config = keywordPreprocessor.trainer_config

    model = Sequential(
        [
            Input(shape=(train_test_data.dimension,)),
            Dense(128, activation="relu"),
            Dropout(trainer_config["dropout"]),
            Dense(train_test_data.n_labels, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=trainer_config["learning_rate"]),
        loss=focal_loss(
            gamma=trainer_config["fl_gamma"],
            alpha=trainer_config["fl_alpha"],
        ),
        metrics=["accuracy", "precision", "recall"],
    )

    model.summary()

    epoch = trainer_config["epoch"]
    batch_size = trainer_config["batch_size"]

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=trainer_config["early_stopping_patience"],
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=trainer_config["reduce_lr_patience"],
        min_lr=1e-6,
    )
    with mlflow.start_run():
        trainer_params = keywordPreprocessor.trainer_config
        mlflow.log_params(trainer_params)

        history = model.fit(
            train_test_data.X_train,
            train_test_data.Y_train,
            epochs=epoch,
            batch_size=batch_size,
            class_weight=train_test_data.label_weight_dict,
            validation_split=trainer_config["validation_split"],
            callbacks=[early_stopping, reduce_lr],
        )

        model_file_path = (
            keywordPreprocessor.config.base_dir
            / "resources"
            / KEYWORD_FOLDER
            / model_name
        ).with_suffix(".keras")
        # make sure folder exist
        model_file_path.parent.mkdir(parents=True, exist_ok=True)

        model.save(model_file_path)
        return model, history


def replace_with_column_names(row: pd.Series, column_names: Dict) -> List[str]:
    """
    Transform a row of binary values and returns a list of column names for which the value in the row is 1.
    Input:
        row: pd.Series. A row of binary values indicating presence (1) or absence (0) of each label.
        column_names: Dict. The anonymous predefined label set.
    Output:
        List[str]: The predicted keywords
    """
    return [column_names[i] for i, value in enumerate(row) if value == 1]


def get_predicted_keywords(prediction: np.ndarray, labels: Dict):
    """
    Convert binary predictions to textual keywords.
    Input:
        prediction: np.ndarray. The predicted binary matrix.
        labels: Dict. The predefined keywords.
    Output:
        predicted_keywords: pd.Series. The predicted keywords for the given targets.
    """
    target_predicted = pd.DataFrame(prediction, columns=list(labels))
    predicted_keywords = target_predicted.apply(
        lambda row: replace_with_column_names(row, labels), axis=1
    )
    if len(predicted_keywords) == 1:
        predicted_keywords = [item.to_json() for item in predicted_keywords[0]]
    return predicted_keywords


def load_saved_model(trained_model: str) -> Optional[load_model]:
    """
    Load a saved pretrained model from file, via a model name
    Input:
        trained_model: str. The name of the trained model file (without extension), located in the `data_discovery_ai/output/` directory.
    Output:
        Optional[keras_load_model]: The loaded Keras model if successful, otherwise `None`.
    """
    model_file_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / KEYWORD_FOLDER
        / trained_model
    ).with_suffix(".keras")
    try:
        saved_model = load_model(model_file_path, compile=False)
        return saved_model
    except (OSError, ValueError, FileNotFoundError) as e:
        logger.error(
            f"Failed to load selected model {trained_model} from folder data_discovery_ai/resources"
        )
        return None
