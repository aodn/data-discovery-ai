# The data delivery mode filter model to classify the metadata records based on their titles, abstracts, and lineages.
# Possible classes are 'Real Time', 'Delayed', and 'Other'.
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
)
import numpy as np
from pathlib import Path
import mlflow  # type: ignore
from typing import Any
from dataclasses import asdict

from sklearn.tree import DecisionTreeClassifier

from data_discovery_ai.config.constants import FILTER_FOLDER
from data_discovery_ai.ml.preprocessor import DeliveryPreprocessor
from data_discovery_ai.utils.agent_tools import (
    save_to_file,
    load_from_file,
    get_text_embedding,
)
from data_discovery_ai import logger


mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)


def train_delivery_model(
    model_name: str, delivery_preprocessor: DeliveryPreprocessor
) -> Any:
    """
    The classification model for predicting the data delivery mode of metadata records, based on their titles, abstracts, and lineages.
    Currently, we applied a self-training model with a random forest classifier as the base model. It extends the idea of semi-supervised learning, in which both
    Input:
        model_name: str. The name of the model, which should be stricter within the options of `AVAILABLE_MODELS` in `data_discovery_ai/common/constants.py`.
        X_labelled_train: np.ndarray. The training data of the metadata records, which is split from the labelled data.
        y_labelled_train: np.ndarray. The labels of the training data, which is split from the labelled data.
        params: ConfigParser. The configuration parameters for the model, which is loaded from the `MODEL_CONFIG` defined in `data_discovery_ai/common/constants.py`.
    Output:
        Any: the classification model for predicting the data delivery mode of metadata records.
    """
    trainer_config = delivery_preprocessor.trainer_config
    threshold = trainer_config.threshold
    train_test_data = delivery_preprocessor.train_test_data
    max_depth = trainer_config.max_depth
    max_leaf_nodes = trainer_config.max_leaf_nodes
    k_best = trainer_config.k_best
    max_iter = trainer_config.max_iter

    base_model = DecisionTreeClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        class_weight="balanced",
        random_state=42,
    )

    # self-training classifier
    self_training_model = SelfTrainingClassifier(
        base_model, threshold=threshold, k_best=k_best, verbose=True, max_iter=max_iter
    )

    mlflow.set_experiment("delivery model experiment")
    mlflow_config = delivery_preprocessor.mlflow_config
    port_num = mlflow_config.port
    mlflow.set_tracking_uri("http://localhost:{}".format(port_num))
    with mlflow.start_run():
        self_training_model.fit(
            train_test_data.X_combined_train, train_test_data.Y_combined_train
        )
        mlflow.log_params(asdict(trainer_config))

        evaluate_model(
            self_training_model, train_test_data.X_test, train_test_data.Y_test
        )

        # model file path
        model_file_path = (
            Path(__file__).resolve().parent.parent
            / "resources"
            / FILTER_FOLDER
            / model_name
        )

        # make sure path exists
        model_file_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_file(self_training_model, model_file_path.with_suffix(".pkl"))

        return self_training_model


def load_saved_model(model_name: str) -> Any:
    """
    Load the saved model from local pickle file.
    The fine name is given by the 'selected_model' from the API request. The model file is end up with ".pkl" suffix.
    Input:
        model_name: str. The name of the model, which should be stricter within the options of `AVAILABLE_MODELS` in `data_discovery_ai/common/constants.py`.
    Output:
        Any: the saved pre-trained model
    """
    # load model pickle file
    model_file_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / FILTER_FOLDER
        / model_name
    )
    trained_model = load_from_file(model_file_path.with_suffix(".pkl"))

    return trained_model


def evaluate_model(model: Any, X_test: np.ndarray, Y_test: np.ndarray) -> None:
    """
    Evaluate the model with the testing data. The evaluation metrics comes from the classification report, which is printed out in the log.
    Input:
        model: Any. The trained model.
        X_test: np.ndarray. The testing data of the metadata records, which is splited from the labelled data.
        Y_test: np.ndarray. The real class ("real-time" or "delayed") of the testing data, which is splited from the labelled data. This can be used as the groundtruth to evaluate the model.
    """
    Y_pred = model.predict(X_test)

    # Generate classification metrics
    acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average="weighted", zero_division=0)
    rec = recall_score(Y_test, Y_pred, average="weighted", zero_division=0)

    # Log evaluation metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", rec)

    report = classification_report(Y_test, Y_pred)
    logger.info(f"Classification report: \n{report}")


def make_prediction(model: Any, description: str) -> np.ndarray:
    """
    Make prediction for a given metadata record, the description is the combination of its title, abstract, and lineage.
    Input:
        model: Any. The trained model.
        description: str. The textual description of the metadata record, which is the combination of its title, abstract, and lineage.
    Output:
        np.ndarray. Return a np.ndarray of size 1, which is the predicted class of the metadata record. This prediction task has only two classes: 0 for "Real-Time" and 1 for "Delayed".
    """
    description_embedding = get_text_embedding(description)
    dimension = description_embedding.shape[0]
    target_X = description_embedding.reshape(1, dimension)

    y_pred = model.predict(target_X)
    return y_pred[0]


def get_predicted_class_name(predicted_class: int) -> str:
    """
    Conver the numeric class to the textual class name
    Input:
        predicted_class: int. The predicted class of the metadata record. It can be 0 or 1.
    Output:
        str. The textual class name of the predicted class. It can be "Real-Time" or "Delayed".
    """
    class_map = {0: "Real-Time", 1: "Delayed", 2: "Other"}
    pred_class = class_map.get(predicted_class)
    return pred_class
