"""
    The module to preprocess the data and prepare datasets for training and evaluating a ML model
"""

import pandas as pd
import ast
import numpy as np
import configparser
from typing import Any, List, Tuple, Union, Optional

from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
from typing import Dict

from data_discovery_ai import logger

# hide warning information from transformers
from transformers import logging as tf_logging

tf_logging.set_verbosity_error()


# data preprocessor for data delivery mode filter model
def identify_ddm_sample(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies a sample set (labelled data) from raw data based on specified vocabulary terms. This function filters raw data (obtained from an Elasticsearch search result).
    This function filters raw data (obtained from an Elasticsearch search result) to identify records with 'onGoing' status.
    Input:
        raw_data: pd.DataFrame. The search result DataFrame from Elasticsearch, expected to contain fields '_id', '_source.title', '_source.description', '_source.summaries.statement' and '_source.summaries.status'.
    Output:
        preprocessed_data: pd.DataFrame. The preprocessed DataFrame containing the necessary information for the data delivery mode filter model with one more column 'information'. The 'information' column is the combination of 'title', 'description' and 'lineage'.
    """  # only keep selected columns
    columns = [
        "_id",
        "_source.title",
        "_source.description",
        "_source.summaries.statement",
        "_source.summaries.status",
    ]
    preprocessed_data = raw_data[columns].copy()

    # change column names
    preprocessed_data.columns = ["id", "title", "abstract", "lineage", "status"]

    # fill na with empty string in lineage column
    preprocessed_data["lineage"].fillna("", inplace=True)

    # add information column, which is the text of title, description and lineage
    preprocessed_data["information"] = (
        preprocessed_data["title"]
        + " [SEP] "
        + preprocessed_data["abstract"]
        + " [SEP] "
        + preprocessed_data["lineage"]
    )

    # only focus on onGoing records
    preprocessed_data = preprocessed_data[preprocessed_data["status"] == "onGoing"]

    return preprocessed_data


def prepare_train_test_ddm(
    data_with_mode: pd.DataFrame, params: configparser.ConfigParser
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the training and testing datasets for the data delivery mode filter model.
    Input:
        data_with_mode: pd.DataFrame. The final data set that contains both the labelled and unlabelled records. It is expected to have these fields: "id", "title", "abstract", "lineage", "status", "information", "embedding", "mode".
    Output:
        X_labelled_train: np.ndarray. The training feature set for labelled data.
        y_labelled_train: np.ndarray. The training target set for labelled data.
        X_combined_train: np.ndarray. The combined training feature set for both labelled and unlabelled data.
        y_combined_train: np.ndarray. The combined training target set for both labelled and unlabelled data.
        X_val: np.ndarray. The testing feature set.
        y_val: np.ndarray. The testing target set.
    """
    # split the data into labelled and unlabelled sets
    labelled_data = data_with_mode[data_with_mode["mode"] != -1]
    logger.info(f"Size of labelled set: {len(labelled_data)}")

    unlabelled_data = data_with_mode[data_with_mode["mode"] == -1]
    logger.info(f"Size of unlabelled set: {len(unlabelled_data)}")

    # only keep embedding column as feature X and mode column as target y for labelled data
    X_labelled = labelled_data["embedding"].tolist()
    y_labelled = labelled_data["mode"].tolist()

    # split labelled data into training and testing sets for validation
    test_size = params.getfloat("filterPreprocessor", "test_size")
    X_labelled_train, X_val, y_train, y_val = train_test_split(
        X_labelled, y_labelled, test_size=test_size, random_state=42
    )
    logger.info(
        f"Size of training set: {len(X_labelled_train)} \n Size of test set: {len(X_val)}"
    )

    # only keep embedding column as feature X and mode column as target y for unlabelled data
    X_unlabelled = unlabelled_data["embedding"].tolist()
    y_unlabelled = unlabelled_data["mode"].tolist()

    # combine unlabelled data with labelled training data for training
    X_combined_train = np.vstack([X_labelled_train, X_unlabelled])
    y_combined_train = np.hstack([y_train, y_unlabelled])
    logger.info(f"size of final training set: {len(X_combined_train)}")

    # just to make sure X and y are same size
    if len(X_combined_train) != len(y_combined_train):
        raise ValueError("X and y are not the same size")

    # just to make sure train and test sets have same dimension
    if len(X_combined_train[0]) != len(X_val[0]):
        raise ValueError("Train and test sets have different dimensions")

    return (
        np.array(X_labelled_train),
        np.array(y_train),
        X_combined_train,
        y_combined_train,
        np.array(X_val),
        np.array(y_val),
    )


def identify_km_sample(raw_data: pd.DataFrame, vocabs: List[str]) -> pd.DataFrame:
    """
    Identifies a sample set from raw data based on specified vocabulary terms. This function filters raw data (obtained from an Elasticsearch search result) to identify records containing specific vocabulary terms in the `keywords` field.

    Input:
        raw_data: pd.DataFrame. The search result DataFrame from Elasticsearch, expected to contain fields `_id`, `_source.title`, `_source.description`, and `_source.themes`.
        vocabs: List[str]. A list of vocabulary terms to search for within each entry's `keywords` field. predefined in common/keyword_classification_parameters.json file.
    Output:
        sampleSet: pd.Dataframe. A DataFrame representing the identified sample set, containing only entries with keywords matching the specified vocabularies.
    """
    raw_data_cleaned = raw_data[
        ["_id", "_source.title", "_source.description", "_source.themes"]
    ]
    raw_data_cleaned.columns = ["id", "title", "description", "keywords"]
    sampleSet = raw_data_cleaned[
        raw_data_cleaned["keywords"].apply(
            lambda terms: any(
                any(vocab in k["title"] for vocab in vocabs) for k in terms
            )
        )
    ]
    return sampleSet


def sample_preprocessor(sampleSet: pd.DataFrame, vocabs: List[str]) -> pd.DataFrame:
    """
    Preprocess sample set data, including extract and reformat labels, and remove empty value records
    Input:
        sampleSet: pd.Dataframe. The identified sample set
        vocabs: List[str]. A list of vocabulary names, the predefined vocabularies
    Output:
        sampleSet: pd.Dataframe. The sample set with filtered keywords
    """

    sampleSet["keywords"] = sampleSet["keywords"].apply(
        lambda x: keywords_formatter(x, vocabs)
    )
    sampleSet["information"] = sampleSet["title"] + " [SEP] " + sampleSet["description"]
    return sampleSet


def identify_rare_labels(Y_df: pd.DataFrame, threshold: int, labels: Dict) -> List[int]:
    """
    Identify rare labels under a threshold.
    Input:
        Y_df: pd.Dataframe. The target variables for all items in the sample set.
        threshold: int. The threshold for identifing rare labels, if the number of apperance is under this threshold, the label is considered as a rare label.
        labels: Dict. The predefined label set which contains all labels
    Output:
        rare_label_index: List[int]. The indexes of rare labels in Y
    """
    label_distribution = Y_df.copy()
    label_distribution = label_distribution.sum()
    label_distribution.sort_values()
    label_distribution_df = label_distribution.to_frame(name="count")
    rare_labels = label_distribution_df[
        label_distribution_df["count"] <= threshold
    ].index.to_list()
    return rare_labels


def get_description_embedding(text: str) -> np.ndarray:
    """
    Calculates the embedding of a given text using a pre-trained BERT model. This function tokenizes the input text, processes it through a BERT model, and extracts the CLS token embedding, which serves as a representation of the text.
    Input:
        text: str. A piece of textual information. In the context of keyword classification task, this is the abstract of a metadata record.
    Output:
        text_embedding: np.ndarray. A numpy array representing the text embedding as a feature vector.
    """
    # https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/bert#transformers.TFBertModel
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # use in Tensorflow https://huggingface.co/google-bert/bert-base-uncased
    model = TFBertModel.from_pretrained("bert-base-uncased")

    # https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.return_tensors, set as 'tf' to return tensorflow tensor
    inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True)
    outputs = model(inputs)
    text_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    # output as a 1D array, shape (768,)
    return text_embedding.squeeze()


def prepare_train_test(
    X: np.ndarray, Y: np.ndarray, params: configparser.ConfigParser
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the training and testing datasets using multi-label stratified splitting.
    This function splits the feature matrix X and target matrix Y into training and testing sets based on parameters for multi-label stratified shuffling. It prints dataset information and returns the dimensions, number of labels, and split data for training and testing.
    Input:
        X: np.ndarray. Feature matrix of shape (n_samples, dimension).
        Y: np.ndarray. Target matrix of shape (n_samples, n_labels).
        params: Dict[str, Any].
    Output:
        Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - dim: int. Feature dimension of X.
            - n_labels: int. Number of labels in Y.
            - X_train: np.ndarray. Training features.
            - Y_train: np.ndarray. Training targets.
            - X_test: np.ndarray. Testing features.
            - Y_test: np.ndarray. Testing targets.
    """
    # get X, Y shape
    n_labels = Y.shape[1]
    dim = X.shape[1]

    n_splits = params.getint("keywordPreprocessor", "n_splits")
    test_size = params.getfloat("keywordPreprocessor", "test_size")
    train_test_random_state = params.getint(
        "keywordPreprocessor", "train_test_random_state"
    )
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=train_test_random_state
    )

    for train_index, test_index in msss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Dimension: {dim}")
    logger.info(f"No. of labels: {n_labels}")
    logger.info(
        f"Train set size: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.2f}%)"
    )
    logger.info(
        f"Test set size: {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.2f}%)"
    )

    return dim, n_labels, X_train, Y_train, X_test, Y_test
