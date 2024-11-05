"""
    The keyword classification model used to identify the potential keywords for non-categorised records.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import logging
from typing import Dict, Callable, Any, Tuple, Optional, List
import os
from pathlib import Path

# Base directory where your Poetry project's pyproject.toml is located
BASE_DIR = Path(__file__).resolve().parent.parent
SUB_DIR = BASE_DIR / "resources"

os.environ["TF_USE_LEGACY_KERAS"] = "1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_class_weights(Y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate label weights by the frequency of a label appears in all records
    Input:
        Y_train: numpy.ndarray. The train set of Y
    Output:
        label_weight_dic: Dict[int, float]. The label weights, keys are the indexs of labels and values are the weights.
    """
    label_frequency = np.sum(Y_train, axis=0)
    epsilon = 1e-6
    label_weights = np.minimum(1, 1 / (label_frequency + epsilon))

    label_weight_dict = {i: label_weights[i] for i in range(len(label_weights))}
    return label_weight_dict


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
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = -alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)

    return focal_loss_fixed


def keyword_model(
    model_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    class_weight: Dict[int, float],
    dim: int,
    n_labels: int,
    params: Dict[str, Any],
) -> Tuple[Sequential, Any, str]:
    """
    Builds, trains, and evaluates a multi-label classification model for keyword prediction. Train neural network model with configurable hyperparameters (through `common/keyword_classification_parameters.json`), compiles it with a focal loss function, and trains it on the provided training data.
    It also saves the trained model and evaluates it on test data. The saved model can called by the keywordClassifier API service.
    Input:
        model_name: str. The file name which saves the model.
        X_train: np.ndarray. The training feature matrix.
        Y_train: np.ndarray. The training target matrix.
        X_test: np.ndarray. The test feature matrix.
        Y_test: np.ndarray. The test target matrix.
        class_weight: Dict[int, float]. Class weights for handling class imbalance. Calculated via function get_class_weights
        dim: int. The input feature dimension for the model.
        n_labels: int. The number of output labels for the model.
        params: Dict[str, Any]: A dictionary of hyperparameters, including:
            "dropout": float. Dropout rate for regularization.
            "learning_rate": float. Learning rate for the Adam optimizer.
            "fl_gamma: float. Gamma parameter for focal loss.
            "fl_alpha": float. Alpha parameter for focal loss.
            "epoch": int. Number of training epochs.
            "batch": int. Batch size for training.
            "early_stopping_patience": int. Patience for early stopping.
            "reduce_lr_patience": int. Patience for reducing learning rate.
            "validation_split": float. Fraction of data for validation during training.
    Output:
        model, history: Tuple[Sequential, Any]. The trained Keras model and the training history.
    """
    model = Sequential(
        [
            Input(shape=(dim,)),
            Dense(128, activation="relu"),
            Dropout(params.getfloat("keywordModel", "dropout")),
            # Dense(64, activation='relu'),
            # Dropout(params["keywordModel"]["dropout"]),
            Dense(n_labels, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=params.getfloat("keywordModel", "learning_rate")),
        loss=focal_loss(
            gamma=params.getint("keywordModel", "fl_gamma"),
            alpha=params.getfloat("keywordModel", "fl_alpha"),
        ),
        metrics=["accuracy", "precision", "recall"],
    )

    model.summary()

    epoch = params.getint("keywordModel", "epoch")
    batch_size = params.getint("keywordModel", "batch")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=params.getint("keywordModel", "early_stopping_patience"),
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=params.getint("keywordModel", "reduce_lr_patience"),
        min_lr=1e-6,
    )

    history = model.fit(
        X_train,
        Y_train,
        epochs=epoch,
        batch_size=batch_size,
        class_weight=class_weight,
        validation_split=params.getfloat("keywordModel", "validation_split"),
        callbacks=[early_stopping, reduce_lr],
    )
    model_file_path = (SUB_DIR / model_name).with_suffix(".keras")
    # make sure folder exist
    model_file_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the folder exists

    model.save(model_file_path)

    model.evaluate(X_test, Y_test)
    return model, history, model_name


def evaluation(Y_test: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the predicted labels via trained model with test set. The metrics computed are accuracy, Hamming loss, precision, recall, F1 score, and Jaccard index.
    Input:
        Y_test: np.ndarray. The true labels from test set.
        predictions: np.ndarray. The predicted labels from the trained model.
    Output:
        Dict[str, float]: A dictionary containing the calculated evaluation metrics, including precision, recall, F1 score, Hamming loss, Jaccard index, and accuracy.
    """
    accuracy = accuracy_score(Y_test, predictions)
    hammingloss = hamming_loss(Y_test, predictions)
    precision = precision_score(Y_test, predictions, average="micro")
    recall = recall_score(Y_test, predictions, average="micro")
    f1 = f1_score(Y_test, predictions, average="micro")
    jaccard = jaccard_score(Y_test, predictions, average="samples")

    print(f" {precision:.4f} | {recall:.4f} | {f1:.4f} | {hammingloss:.4f} |")
    return {
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1": f"{f1:.4f}",
        "hammingloss": f"{hammingloss:.4f}",
        "Jaccard Index": f"{jaccard:.4f}",
        "accuracy": f"{accuracy:.4f}",
    }


def prediction(X: np.ndarray, model: Any, confidence: float, top_N: int) -> np.ndarray:
    """
    Apply the trained model to generate predictions for the input data X. It uses a confidence threshold to determine predicted labels, marking as 1 for labels with probabilities above the confidence level. If no labels are above the threshold for a sample, the function selects the top N highest probabilities, marking them as positive labels (value 1).
    Input:
        X: np.ndarray. The input feature X matrix used for making predictions.
        model: Any. The trained model (baseline model or Sequential model).
        confidence: float. In the range of [0,1]. The confidence threshold for assigning labels. Predictions above this value are marked as 1.
        top_N: The number of top predictions to select if no predictions meet the confidence threshold.
    Output:
        predicted_labels: np.ndarray. A binary matrix of predicted labels, where each row corresponds to a sample and each column to a label.
    """
    predictions = model.predict(X)
    predicted_labels = (predictions > confidence).astype(int)

    for i in range(predicted_labels.shape[0]):
        if predicted_labels[i].sum() == 0:
            top_indices = np.argsort(predictions[i])[-top_N:]
            predicted_labels[i][top_indices] = 1
    return predicted_labels


def replace_with_column_names(
    row: pd.SparseDtype, column_names: List[str]
) -> List[str]:
    """
    Transform a row of binary values and returns a string of column names (separated by " | ") for which the value in the row is 1.
    Input:
        row: pd.Series. A row of binary values indicating presence (1) or absence (0) of each label.
        column_names: List[str]. The predefiend label set.
    Output:
        str: The predicted keywords, separated by " | "
    """
    return " | ".join([column_names[i] for i, value in enumerate(row) if value == 1])


def get_predicted_keywords(prediction: np.ndarray, labels: List[str]):
    """
    Convert binary predictions to textual keywords.
    Input:
        prediction: np.ndarray. The predicted binary matrix.
        labels: List[str]. The predefiend keywords.
    Output:
        predicted_keywords: pd.Series. The predicted ketwords for the given targets.
    """
    target_predicted = pd.DataFrame(prediction, columns=labels)
    predicted_keywords = target_predicted.apply(
        lambda row: replace_with_column_names(row, labels), axis=1
    )
    return predicted_keywords


def baseline(
    X_train: np.ndarray, Y_train: np.ndarray, model: str
) -> MultiOutputClassifier:
    """
    Trains a baseline multi-output classification model based on the specified algorithm (KNN or DT).
    Input:
        X_train: np.ndarray. The training feature matrix.
        Y_train: np.ndarray. The training target matrix.
        model:str. The type of baseline model to train. Options include:
            - "KNN" for K-Nearest Neighbors.
            - "DT" for Decision Tree.
    Output:
        baseline_model: MultiOutputClassifier. The trained baseline model.
    """
    if model == "KNN":
        baseline_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
        baseline_model.fit(X_train, Y_train)
    elif model == "DT":
        baseModel = DecisionTreeClassifier(random_state=42)
        baseline_model = MultiOutputClassifier(baseModel)
        baseline_model.fit(X_train, Y_train)
    # TODO: add more baseline models
    else:
        raise ValueError(
            f"Unsupported model type: {model}. Please choose 'KNN' or 'DT'."
        )

    return baseline_model


def load_saved_model(trained_model: str) -> Optional[load_model]:
    """
    Load a saved pretrained model from file, via a model name
    Input:
        trained_model: str. The name of the trained model file (without extension), located in the `data_discovery_ai/output/` directory.
    Output:
        Optional[keras_load_model]: The loaded Keras model if successful, otherwise `None`.
    """
    model_file_path = (SUB_DIR / trained_model).with_suffix(".keras")
    try:
        saved_model = load_model(model_file_path, compile=False)
        return saved_model
    except Exception as e:
        print(e)
        logger.info(
            f"Failed to load selected model {trained_model} from folder data_discovery_ai/resources"
        )
        return None
