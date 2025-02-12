# The data delivery mode filter model to classify the metadata records based on their titles, abstracts, and lineages.
# Possible classes are 'Real Time', 'Delayed', and 'Other'.
import logging
import os
from data_discovery_ai.utils.preprocessor import get_description_embedding
import tensorflow as tf
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from pathlib import Path

from typing import Dict, Any
from configparser import ConfigParser

from data_discovery_ai.common.constants import FILTER_FOLDER
from data_discovery_ai.utils.preprocessor import save_to_file, load_from_file, get_description_embedding

os.environ["TF_USE_LEGACY_KERAS"] = "1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Delete this line after fix 'module not exist issue' in notebooks
FILTER_FOLDER = "DataDeliveryModeFilter"

def ddm_filter_model(
    model_name: str,
    X_labelled_train: np.ndarray,
    y_labelled_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: ConfigParser
) -> Any:
    """
    TODO: add description
    """
    n_estimators = params.getint("filterModel", "n_estimators")
    random_state = params.getint("filterModel", "random_state")
    threshold = params.getfloat("filterModel", "threshold")

    base_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    # self-training classifier
    self_training_model = SelfTrainingClassifier(base_model, threshold=threshold)
    self_training_model.fit(X_labelled_train, y_labelled_train)

    # model file path
    model_file_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / FILTER_FOLDER
        / model_name
    ).with_suffix(".pkl")

    # make sure path exists
    model_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_file(self_training_model, model_file_path)

    return self_training_model

def load_saved_model(model_name: str) -> Any:
    """
        TODO: add description
    """
    # load pickle file
    model_file_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / FILTER_FOLDER
        / model_name
    ).with_suffix(".pkl")
    trained_model = load_from_file(model_file_path)

    return trained_model


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
        TODO: add description
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification report: \n{report}")
    print(f"Classification report: \n{report}")


def make_prediction(
        model: Any,
        description: str
) -> np.ndarray:
    """
        TODO: add description
    """
    description_embedding = get_description_embedding(description)
    dimension = description_embedding.shape[0]
    target_X = description_embedding.reshape(1, dimension)

    y_pred = model.predict(target_X)
    return y_pred[0]


def get_predicted_class_name(predicted_class: int) -> str:
    """
        TODO: add description
    """
    class_map = {0: "Real Time", 1: "Delayed", 2: "Other"}
    pred_class = class_map.get(predicted_class)
    return pred_class