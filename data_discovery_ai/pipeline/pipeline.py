# ML pipeline, need to be deployed to prefect, etc in the future
from data_discovery_ai.utils.preprocessor import *
from data_discovery_ai.utils import preprocessor as preprocessor

import data_discovery_ai.model.keywordModel as keyword_model
import data_discovery_ai.model.filteringModel as ddm_model
import data_discovery_ai.utils.es_connector as connector
import data_discovery_ai.service.keywordClassifier as classifier
from data_discovery_ai.utils.config_utils import ConfigUtil
from data_discovery_ai.common.constants import *

import sys

sys.modules["preprocessor"] = preprocessor

import numpy as np
import pandas as pd
from typing import Any, Dict
from dataclasses import dataclass
import tempfile
import os

from data_discovery_ai import logger


@dataclass
class TrainTestData:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    label_weight_dict: Dict[int, float]
    dimension: int
    n_labels: int


@dataclass
class DDMTrainTestData:
    X_labelled_train: np.ndarray
    y_labelled_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_combined_train: np.ndarray
    y_combined_train: np.ndarray


class BasePipeline:
    def __init__(
        self, is_data_changed: bool, use_pretrained_model: bool, model_name: str
    ):
        self.config = ConfigUtil()
        self.is_data_changed = is_data_changed
        self.use_pretrained_model = use_pretrained_model
        self.model_name = model_name
        # validate model name with accepted values, defined in data_discovery_ai/common/constants.py
        if not self.is_valid_model():
            raise ValueError(
                'Available model name: ["development", "staging", "production", "experimental", "benchmark"]'
            )

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
        # get ES_INDEX_NAME from environment
        index = os.getenv("ES_INDEX_NAME", default=ES_INDEX_NAME)
        raw_data = connector.search_es(
            client=client, index=index, batch_size=BATCH_SIZE, sleep_time=SLEEP_TIME
        )
        return raw_data


class DataDeliveryModeFilterPipeline(BasePipeline):
    def __init__(
        self, is_data_changed: bool, use_pretrained_model: bool, model_name: str
    ) -> None:
        super().__init__(
            is_data_changed=is_data_changed,
            use_pretrained_model=use_pretrained_model,
            model_name=model_name,
        )

        # create temp folder
        self.temp_dir = tempfile.mkdtemp()

        self.params = self.config.load_model_config()

        # define predicted labels
        self.predicted_class = None

    # extends the fetch_raw_data method from BasePipeline
    def fetch_raw_data(self) -> pd.DataFrame:
        return super().fetch_raw_data()

    def prepare_train_test_sets(self, preprocessed_data: pd.DataFrame) -> TrainTestData:
        """
        Prepare the training and testing tests from the preprocessed data.
        Input:
            preprocessed_data: pd.DataFrame. The preprocessed data from the raw data which filters the OnGoing records and calculates the embeddings.
        Output:
            train_test_data: TrainTestData. The training and testing data for the model follows a customised dataclass object.

        """
        # label the data: we have a predefined rule to label the OnGoing data
        label_ddm_sample = preprocessor.label_ddm_sample(preprocessed_data)

        # prepare train and test sets.
        (
            X_labelled_train,
            y_labelled_train,
            X_combined_train,
            y_combined_train,
            X_test,
            y_test,
        ) = preprocessor.prepare_train_test_ddm(label_ddm_sample, self.params)

        # pack the results into a Dataclass object
        train_test_data = DDMTrainTestData(
            X_labelled_train=X_labelled_train,
            y_labelled_train=y_labelled_train,
            X_test=X_test,
            y_test=y_test,
            X_combined_train=X_combined_train,
            y_combined_train=y_combined_train,
        )
        return train_test_data

    def make_prediction(self, trained_model: Any, description: str, pca: Any) -> None:
        """
        Make a prediction on the given description using a trained data delivery mode filter model.
        Input:
            trained_model: Any. The trained model which is passed through local saved pickle file, with suffix ".pkl".
            description: str. The textual information for prediction. It is a combined string of title, abstract, and lineage with the seperator "[SEP]". These values are passed through API requests.
            pca: Any. The PCA model which is passed through local saved pickle file, with suffix ".pca.pkl".
        """
        prediction = ddm_model.make_prediction(
            trained_model, description=description, pca=pca
        )
        self.predicted_class = ddm_model.get_predicted_class_name(prediction)

    def pipeline(self, title: str, abstract: str, lineage: str) -> None:
        """
        The data delivery mode filter pipeline. Called by the API request to make prediction with a ML model.
        Input:
            title: str. The title of the metadata record.
            abstract: str. The abstract of the metadata record.
            lineage: str. The lineage of the metadata record.
        """

        # define resource files paths
        base_dir = self.config.base_dif
        full_path = base_dir / "resources" / FILTER_FOLDER / FILTER_PREPROCESSED_FILE

        # concat textual information for prediction
        description = f"{title} [SEP] {abstract} [SEP] {lineage}"

        if self.is_data_changed:
            raw_data = self.fetch_raw_data()

            # only focus on "OnGoing" records with three textual fields: title, abstract, and lineage
            preprocessed_data = preprocessor.identify_ddm_sample(raw_data)
            preprocessed_data_embedding = preprocessor.calculate_embedding(
                preprocessed_data
            )
            # save preprocessed data to resources as calculating embeddings is time-consuming
            preprocessor.save_to_file(preprocessed_data_embedding, full_path)

        else:
            # load preprocessed data from resource
            preprocessed_data = preprocessor.load_from_file(full_path)
            train_test_data = self.prepare_train_test_sets(preprocessed_data)

        # decision-making: use pretrained model or not
        if self.use_pretrained_model:
            # load saved model and its pca from pickle files
            trained_model, pca = ddm_model.load_saved_model(self.model_name)

            # make prediction
            self.make_prediction(
                trained_model=trained_model, description=description, pca=pca
            )
        else:
            # train the model and save to file
            trained_model, pca = ddm_model.ddm_filter_model(
                model_name=self.model_name,
                X_train=train_test_data.X_combined_train,
                y_train=train_test_data.y_combined_train,
                params=self.params,
            )
            ddm_model.evaluate_model(
                trained_model, train_test_data.X_test, train_test_data.y_test, pca
            )

            # make prediction
            self.make_prediction(
                trained_model=trained_model, description=description, pca=pca
            )


class KeywordClassifierPipeline(BasePipeline):
    def __init__(
        self, is_data_changed: bool, use_pretrained_model: bool, model_name: str
    ) -> None:
        """
        Init the pipeline, load parameters from file.
        Input:
            isDataChanged: bool. A flag to show whether the data (metadata records) significantly changed. Set as True if data changed, which means sample set needs to be repreprocessed, as well as the model need to be re-trained.
            usePretrainedModel: bool. Choose whether to use pretrained model or train the model and then to be used. If set as True, the model_name should be given.
            model_name: str. The model name that saved in a .keras file.
        """
        # extends the BasePipeline class
        super().__init__(
            is_data_changed=is_data_changed,
            use_pretrained_model=use_pretrained_model,
            model_name=model_name,
        )
        self.params = self.config.load_model_config()

        # create temp folder
        self.temp_dir = tempfile.mkdtemp()

        # predefine label set for prediction
        self.labels = None

        # define predicted labels
        self.predicted_labels = None

    # extends the fetch_raw_data method from BasePipeline
    def fetch_raw_data(self) -> pd.DataFrame:
        return super().fetch_raw_data()

    def set_labels(self, labels):
        self.labels = labels

    def prepare_sample_set(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares a processed sample set from raw data via filtering, preprocessing and embedding calculations.
        This method executes several processing steps on the raw data:
            1. identify_km_sample: Identifies samples containing specific vocabulary terms from the "vocabs" parameter.
            2. sample_preprocessor: Cleans and preprocesses the identified sample set by reformatting labels and removing empty records.
            3. calculate_embedding: Calculates embeddings for each entry in the preprocessed samples.
            4. Saves the processed sample set to a file, then reloads it for subsequent use.

        Input:
            raw_data: pd.DataFrame. The raw data from Elasticsearch in tabular format.
        Output:
            preprocessed_sampleSet: pd.DataFrame. Representing the processed sample set, with an additional embedding column.
        """
        vocabs = self.params["keywordPreprocessor"]["vocabs"].split(", ")
        labelled_ds = preprocessor.identify_km_sample(raw_data, vocabs)
        preprocessed_samples = preprocessor.sample_preprocessor(labelled_ds, vocabs)

        # drop empty keywords rows
        filtered_sample_set = preprocessed_samples[
            preprocessed_samples["keywords"].apply(lambda x: x != [])
        ]

        sample_set = preprocessor.calculate_embedding(filtered_sample_set)

        full_path = os.path.join(self.temp_dir, KEYWORD_SAMPLE_FILE)

        preprocessor.save_to_file(sample_set, full_path)
        return sample_set

    def prepare_train_test_sets(self, sample_set: pd.DataFrame) -> TrainTestData:
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
        X, Y, Y_df, labels = preprocessor.prepare_X_Y(sample_set)

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
        label_weight_dict = keyword_model.get_class_weights(Y_train)

        # Apply additional oversampling (Random Over Sampling) to the training set
        # TODO: rare_keyword_index needs attention: Expected type 'list[int]', got 'None' instead
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
        trained_model, history, model_name = keyword_model.keyword_model(
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
        predicted_labels = keyword_model.prediction(
            train_test_data.X_test, trained_model, confidence, top_N
        )
        eval_results = keyword_model.evaluation(
            Y_test=train_test_data.Y_test, predictions=predicted_labels
        )
        logger.info(eval_results)

    def make_prediction(self, description: str) -> list[Any]:
        """
        Makes a prediction on the given description using a trained keyword classifier model Generates predicted labels for the given description using the trained keyword
        classifier model specified by self.model_name.
        Input:
            description: str. The textual abstract of a metadata record
        Output:
            predicted_labels: str. The predicted keywords by the trained keyword classifier model
        """
        predicted_labels = classifier.classify_keyword(
            trained_model=self.model_name, description=description, labels=self.labels
        )
        logger.info(predicted_labels)
        self.predicted_labels = predicted_labels
        return predicted_labels

    def pipeline(self, title: str, abstract: str) -> None:
        """
        The keyword classifier pipeline.
        Inputs:
            isDataChanged: bool. The indicator to call the data preprocessing module or not.
            usePretrainedModel: bool. The indicator to use the pretrained model or not.
            title: str. The item title which is used for making prediction.
            abstract: str. The item abstract which is used for making prediction
            selected_model: str. The model name for a selected pretrained model.
        """
        description = f"{title} [SEP] {abstract}"
        # define resource files paths
        base_dir = self.config.base_dif
        full_sample_set_path = (
            base_dir / "resources" / KEYWORD_FOLDER / KEYWORD_SAMPLE_FILE
        )
        full_label_map_path = (
            base_dir / "resources" / KEYWORD_FOLDER / KEYWORD_LABEL_FILE
        )

        # data not changed, so load the preprocessed data from resource
        if not self.is_data_changed:
            sample_set = preprocessor.load_from_file(full_sample_set_path)

            # usePretrainedModel = True
            if self.use_pretrained_model:
                predefined_labels = preprocessor.load_from_file(full_label_map_path)
                self.set_labels(labels=predefined_labels)
            # usePretrainedModel = False
            else:
                # retrain the model
                train_test_data = self.prepare_train_test_sets(sample_set)
                preprocessor.save_to_file(self.labels, full_label_map_path)
                self.train_evaluate_model(train_test_data)

        # data changed, so start from the data preprocessing module
        else:
            raw_data = self.fetch_raw_data()
            sample_set = self.prepare_sample_set(raw_data=raw_data)
            preprocessor.save_to_file(sample_set, full_sample_set_path)

            train_test_data = self.prepare_train_test_sets(sample_set)
            preprocessor.save_to_file(self.labels, full_label_map_path)

            self.train_evaluate_model(train_test_data)
        self.make_prediction(description)
