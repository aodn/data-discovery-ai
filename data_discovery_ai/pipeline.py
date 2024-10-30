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
        params = configparser.ConfigParser()
        params.read("data_discovery_ai/common/keyword_classification_parameters.ini")
        self.params = params
        self.isDataChanged = isDataChanged
        self.usePretrainedModel = usePretrainedModel
        self.model_name = model_name
        self.model = None
        # TODO: needs to define what are the accepted values
        #  validate against the list declared in data_discovery_ai/common/constants.py? needs to be a controlled list of values
        if self.usePretrainedModel and self.model_name is None:
            raise ValueError("model name should be given to use pretrained model")

    def fetch_raw_data(self) -> pd.DataFrame:
        """
        Fetches raw data from Elasticsearch and returns it as a DataFrame.
        Output:
            raw_data: pd.DataFrame. A DataFrame containing the raw data retrieved from Elasticsearch.
        """
        client = connector.connect_es(config_path="./esManager.config")
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
        preprocessor.save_to_file(
            sampleSet, "data_discovery_ai/input/keyword_sample.pkl"
        )
        sampleSet = preprocessor.load_from_file(
            "data_discovery_ai/input/keyword_sample.pkl"
        )
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
        preprocessor.save_to_file(labels, "data_discovery_ai/input/labels.pkl")

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
        logger.info(predicted_labels)
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
            sampleSet = preprocessor.load_from_file(
                "data_discovery_ai/input/keyword_sample.pkl"
            )
        train_test_data = keyword_classifier_pipeline.prepare_train_test_sets(sampleSet)
        keyword_classifier_pipeline.train_evaluate_model(train_test_data)

        keyword_classifier_pipeline.make_prediction(description)


def test():
    item_description = """
                        Ecological and taxonomic surveys of hermatypic scleractinian corals were carried out at approximately 100 sites around Lord Howe Island. Sixty-six of these sites were located on reefs in the lagoon, which extends for two-thirds of the length of the island on the western side. Each survey site consisted of a section of reef surface, which appeared to be topographically and faunistically homogeneous. The dimensions of the sites surveyed were generally of the order of 20m by 20m. Where possible, sites were arranged contiguously along a band up the reef slope and across the flat. The cover of each species was graded on a five-point scale of percentage relative cover. Other site attributes recorded were depth (minimum and maximum corrected to datum), slope (estimated), substrate type, total estimated cover of soft coral and algae (macroscopic and encrusting coralline). Coral data from the lagoon and its reef (66 sites) were used to define a small number of site groups which characterize most of this area.Throughout the survey, corals of taxonomic interest or difficulty were collected, and an extensive photographic record was made to augment survey data. A collection of the full range of form of all coral species was made during the survey and an identified reference series was deposited in the Australian Museum.In addition, less detailed descriptive data pertaining to coral communities and topography were recorded on 12 reconnaissance transects, the authors recording changes seen while being towed behind a boat.
                        The purpose of this study was to describe the corals of Lord Howe Island (the southernmost Indo-Pacific reef) at species and community level using methods that would allow differentiation of community types and allow comparisons with coral communities in other geographic locations.
                        """
    pipeline(
        isDataChanged=False,
        usePretrainedModel=False,
        description=item_description,
        selected_model="test_keyword_pipeline",
    )


if __name__ == "__main__":
    test()
