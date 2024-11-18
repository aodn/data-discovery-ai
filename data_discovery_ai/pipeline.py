from data_discovery_ai.utils.preprocessor import *
from data_discovery_ai.utils import preprocessor as preprocessor

import data_discovery_ai.model.keywordModel as model
import data_discovery_ai.utils.es_connector as connector
import data_discovery_ai.service.keywordClassifier as keywordClassifier
from data_discovery_ai.utils.config_utils import ConfigUtil
from data_discovery_ai.common.constants import (
    AVAILABLE_MODELS,
    KEYWORD_SAMPLE_FILE,
    KEYWORD_LABEL_FILE,
)

import sys

sys.modules["preprocessor"] = preprocessor

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
from dataclasses import dataclass
import logging
import tempfile
import os
import json


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        self.config = ConfigUtil()
        self.params = self.config.load_keyword_config()
        self.isDataChanged = isDataChanged
        self.usePretrainedModel = usePretrainedModel
        # validate model name with accepted values, defined in data_discovery_ai/common/constants.py
        self.model_name = model_name
        if not self.is_valid_model():
            raise ValueError(
                'Available model name: ["development", "staging", "production", "experimental", "benchmark"]'
            )
        # create temp folder
        self.temp_dir = tempfile.mkdtemp()

        # define labels for prediction
        self.labels = None

    def set_labels(self, labels):
        self.labels = labels

    """
        Validate model name within fixed selections
        Input:
            model_name: str. The file name of the saved model. restricted within four options: development, staging, production, and test
    """

    def is_valid_model(self) -> bool:
        valid_model_name = AVAILABLE_MODELS
        self.model_name = self.model_name.lower()
        if self.model_name in valid_model_name:
            return True
        else:
            return False

    def fetch_raw_data(self) -> pd.DataFrame:
        """
        Fetches raw data from Elasticsearch and returns it as a DataFrame.
        Output:
            raw_data: pd.DataFrame. A DataFrame containing the raw data retrieved from Elasticsearch.
        """
        es_config = self.config.load_es_config()

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

        # drop empty keywords rows
        sampleSet_filtered = sampleSet[sampleSet["keywords"].apply(lambda x: x != [])]

        full_path = os.path.join(self.temp_dir, KEYWORD_SAMPLE_FILE)

        preprocessor.save_to_file(sampleSet_filtered, full_path)
        sampleSet_filtered = preprocessor.load_from_file(full_path)
        return sampleSet_filtered

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

        self.labels = labels

        # save labels for pretrained model to use for prediction
        full_path = os.path.join(self.temp_dir, KEYWORD_LABEL_FILE)
        preprocessor.save_to_file(labels, full_path)

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
            trained_model=self.model_name, description=description, labels=self.labels
        )
        print(predicted_labels)
        return predicted_labels


def pipeline(
    isDataChanged: bool, usePretrainedModel: bool, description: str, selected_model: str
) -> None:
    """
    The keyword classifier pipeline.
    Inputs:
        isDataChanged: bool. The indicator to call the data preprocessing module or not.
        usePretrainedModel: bool. The indicator to use the pretrained model or not.
        description: str. The item description which is used for making prediction.
        selected_model: str. The model name for a selected pretrained model.
    """
    keyword_classifier_pipeline = KeywordClassifierPipeline(
        isDataChanged=isDataChanged,
        usePretrainedModel=usePretrainedModel,
        model_name=selected_model,
    )

    # define resource files paths
    base_dir = keyword_classifier_pipeline.config.base_dif
    full_sampleSet_path = base_dir / "resources" / KEYWORD_SAMPLE_FILE
    full_labelMap_path = base_dir / "resources" / KEYWORD_LABEL_FILE

    # data not changed, so load the preprocessed data from resource
    if not isDataChanged:
        sampleSet = preprocessor.load_from_file(full_sampleSet_path)
        predefinedLabels = preprocessor.load_from_file(full_labelMap_path)

        # usePretrainedModel = True
        if keyword_classifier_pipeline.usePretrainedModel:
            keyword_classifier_pipeline.set_labels(labels=predefinedLabels)
        # usePretrainedModel = False
        else:
            # retrain the model
            train_test_data = keyword_classifier_pipeline.prepare_train_test_sets(
                sampleSet
            )
            preprocessor.save_to_file(
                keyword_classifier_pipeline.labels, full_labelMap_path
            )
            keyword_classifier_pipeline.train_evaluate_model(train_test_data)

    # data changed, so start from the data preprocessing module
    else:
        raw_data = keyword_classifier_pipeline.fetch_raw_data()
        sampleSet = keyword_classifier_pipeline.prepare_sampleSet(raw_data=raw_data)
        preprocessor.save_to_file(sampleSet, full_sampleSet_path)
        preprocessor.save_to_file(
            keyword_classifier_pipeline.labels, full_labelMap_path
        )
        train_test_data = keyword_classifier_pipeline.prepare_train_test_sets(sampleSet)
        keyword_classifier_pipeline.train_evaluate_model(train_test_data)
    keyword_classifier_pipeline.make_prediction(description)
