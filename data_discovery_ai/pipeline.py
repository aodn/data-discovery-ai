import data_discovery_ai.utils.preprocessor as preprocessor
import data_discovery_ai.model.keywordModel as model
import data_discovery_ai.utils.es_connector as connector
import data_discovery_ai.service.keywordClassifier as keywordClassifier
import numpy as np
import json
import pandas as pd
import configparser
from typing import Any, Dict, Tuple
from dataclasses import dataclass
import logging
from data_discovery_ai.common.constants import (
    KEYWORD_CONFIG,
    AVAILABLE_MODELS,
    ELASTICSEARCH_CONFIG,
    KEYWORD_SAMPLE_FILE,
    KEYWORD_LABEL_FILE
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from pathlib import Path

# Base directory where your Poetry project's pyproject.toml is located
BASE_DIR = Path(__file__).resolve().parent

# Construct the full path to the .ini file
config_file_path = BASE_DIR / "common" / KEYWORD_CONFIG
if not config_file_path.exists():
    raise FileNotFoundError(
        f"The configuration file was not found at {config_file_path}"
    )

elasticsearch_config_file_path = BASE_DIR / "common" / ELASTICSEARCH_CONFIG
if not config_file_path.exists():
    raise FileNotFoundError(
        f"The configuration file was not found at {config_file_path}"
    )


@dataclass
class TrainTestData:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    label_weight_dict: Dict[int, float]
    dimension: int
    n_labels: int


class KeywordClassifierPipeline:
    def __init__(
        self, isDataChanged: bool, usePretrainedModel: bool, model_name: str
    ) -> None:
        """
        Init the pipeline, load parameters from file.
        Input:
            isDataChanged: bool. A flag to show whether the data (metadata records) significantly changed. Set as True if data changed, which means sample set needs to be repreprocessed, as well as the model need to be re-trained.
            usePretrainedModel: bool. Choose whether to use pretrained model or train the model and then to be used. If set as True, the model_name should be given.
            model_name: str. The model name that saved in a .keras file.
        """
        params = configparser.ConfigParser()
        params.read(config_file_path)
        self.params = params
        self.isDataChanged = isDataChanged
        self.usePretrainedModel = usePretrainedModel
        # validate model name with accepted values, defined in data_discovery_ai/common/constants.py
        self.validate_model_name(model_name=model_name)

    """
        Validate model name within fixed selections
        Input:
            model_name: str. The file name of the saved model. restricted within four options: development, staging, production, and test
    """

    def validate_model_name(self, model_name) -> None:
        valid_model_name = AVAILABLE_MODELS
        if model_name.lower() not in valid_model_name:
            raise ValueError(
                'Available model name: ["development", "staging", "production", "experimental", "benchmark"]'
            )
        else:
            self.model_name = model_name.lower()

    def fetch_raw_data(self) -> pd.DataFrame:
        """
        Fetches raw data from Elasticsearch and returns it as a DataFrame.
        Output:
            raw_data: pd.DataFrame. A DataFrame containing the raw data retrieved from Elasticsearch.
        """
        es_config = configparser.ConfigParser()
        es_config.read(elasticsearch_config_file_path)

        client = connector.connect_es(es_config)
        raw_data = connector.search_es(client)
        return raw_data

    def prepare_sampleSet(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares a processed sample set from raw data via filtering, preprocessing and embedding calculations.
        This method executes several processing steps on the raw data:
            1. identify_sample: Identifies samples containing specific vocabulary terms from the "vocabs" parameter.
            2. sample_preprocessor: Cleans and preprocesses the identified sample set by reformatting labels and removing empty records.
            3. calculate_embedding: Calculates embeddings for each entry in the preprocessed samples.
            4. Saves the processed sample set to a file, then reloads it for subsequent use.

        Input:
            raw_data: pd.DataFrame. The raw data from Elasticsearch in tabular format.
        Output:
            preprocessed_sampleSet: pd.DataFrame. Representing the processed sample set, with an additional embedding column.
        """
        vocabs = self.params["preprocessor"]["vocabs"].split(", ")
        labelledDS = preprocessor.identify_sample(raw_data, vocabs)
        preprocessed_samples = preprocessor.sample_preprocessor(labelledDS, vocabs)
        sampleSet = preprocessor.calculate_embedding(preprocessed_samples)
        preprocessor.save_to_file(sampleSet, "keyword_sample.pkl")
        sampleSet = preprocessor.load_from_file("keyword_sample.pkl")
        return sampleSet

    def prepare_train_test_sets(self, sampleSet: pd.DataFrame) -> TrainTestData:
        """
        Prepares training and test sets from a given sample set by processing features and labels,
        handling rare labels, and applying resampling techniques.

        This method performs the following steps:
        1. Extracts features (X) and labels (Y) from the sample set.
        2. Identifies rare labels based on a threshold, and handles them by custom resampling.
        3. Splits the data into training and test sets and applies oversampling to the training set.
        4. Calculates and returns class weights, dimensionality, and the number of unique labels.

        Input:
            sampleSet: DataFrame containing the sample set, with features and target labels prepared for training.
        Output:
            A customized dataclass TrainTestData containing the following elements:
                - X_train: Training features after oversampling.
                - Y_train: Training labels after oversampling.
                - X_test: Test features.
                - Y_test: Test labels.
                - label_weight_dict: A dictionary of label weights for handling class imbalance.
                - dimension: The dimensionality of the feature set.
                - n_labels: The number of unique labels.
        """

        # Prepare feature matrix (X) and label matrix (Y) from the sample set
        X, Y, Y_df, labels = preprocessor.prepare_X_Y(sampleSet)

        # Save the labels to a file for persistence
        preprocessor.save_to_file(labels, "labels.pkl")

        # Identify rare labels based on a predefined threshold
        rare_label_threshold = self.params.getint(
            "preprocessor", "rare_label_threshold"
        )
        rare_label_index = preprocessor.identify_rare_labels(
            Y_df, rare_label_threshold, labels
        )

        # Apply custom resampling to handle rare labels
        X_oversampled, Y_oversampled = preprocessor.resampling(
            X_train=X, Y_train=Y, strategy="custom", rare_keyword_index=rare_label_index
        )

        # Split data into training and test sets, then apply additional preprocessing
        dim, n_labels, X_train, Y_train, X_test, Y_test = (
            preprocessor.prepare_train_test(X_oversampled, Y_oversampled, self.params)
        )

        # Calculate class weights to manage class imbalance
        label_weight_dict = model.get_class_weights(Y_train)

        # Apply additional oversampling (Random Over Sampling) to the training set
        X_train_oversampled, Y_train_oversampled = preprocessor.resampling(
            X_train=X_train, Y_train=Y_train, strategy="ROS", rare_keyword_index=None
        )

        # pack the result into a customised dataclass object
        train_test_data = TrainTestData(
            X_train=X_train_oversampled,
            Y_train=Y_train_oversampled,
            X_test=X_test,
            Y_test=Y_test,
            label_weight_dict=label_weight_dict,
            dimension=dim,
            n_labels=n_labels,
        )
        return train_test_data

    def train_evaluate_model(self, train_test_data: TrainTestData) -> None:
        """
        Trains and evaluates the keyword classifier model using the provided training and test data. Calculates
        evaluation metrics based on model predictions and the actual test labels.

        Input:
            train_test_data: An instance of TrainTestData containing training and test data, label weights, feature dimensions, and other necessary information.
        """
        # train keyword model
        trained_model, history, model_name = model.keyword_model(
            model_name=self.model_name,
            X_train=train_test_data.X_train,
            Y_train=train_test_data.Y_train,
            X_test=train_test_data.X_test,
            Y_test=train_test_data.Y_test,
            class_weight=train_test_data.label_weight_dict,
            dim=train_test_data.dimension,
            n_labels=train_test_data.n_labels,
            params=self.params,
        )
        # evaluate
        confidence = self.params.getfloat("keywordModel", "confidence")
        top_N = self.params.getint("keywordModel", "top_N")
        predicted_labels = model.prediction(
            train_test_data.X_test, trained_model, confidence, top_N
        )
        eval = model.evaluation(
            Y_test=train_test_data.Y_test, predictions=predicted_labels
        )
        print(eval)

    def make_prediction(self, description: str) -> str:
        """
        Makes a prediction on the given description using a trained keyword classifier model Generates predicted labels for the given description using the trained keyword
        classifier model specified by self.model_name.
        Input:
            description: str. The textual abstract of a metadata record
        Output:
            predicted_labels: str. The predicted keywords by the trained keyword classifier model
        """
        predicted_labels = keywordClassifier.keywordClassifier(
            trained_model=self.model_name, description=description
        )
        print(predicted_labels)
        return predicted_labels


def pipeline(isDataChanged, usePretrainedModel, description, selected_model):
    keyword_classifier_pipeline = KeywordClassifierPipeline(
        isDataChanged=isDataChanged,
        usePretrainedModel=usePretrainedModel,
        model_name=selected_model,
    )
    if keyword_classifier_pipeline.usePretrainedModel:
        keyword_classifier_pipeline.make_prediction(description)
    else:
        if keyword_classifier_pipeline.isDataChanged:
            raw_data = keyword_classifier_pipeline.fetch_raw_data()
            sampleSet = keyword_classifier_pipeline.prepare_sampleSet(raw_data=raw_data)
        else:
            sampleSet = preprocessor.load_from_file(KEYWORD_SAMPLE_FILE)
        train_test_data = keyword_classifier_pipeline.prepare_train_test_sets(sampleSet)
        keyword_classifier_pipeline.train_evaluate_model(train_test_data)

        keyword_classifier_pipeline.make_prediction(description)
