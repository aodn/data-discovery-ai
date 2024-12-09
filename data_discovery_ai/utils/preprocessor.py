"""
    The module to preprocess the data and prepare datasets for training and evaluating a ML model
"""

import logging
import pickle
import pandas as pd
import ast
import os
import numpy as np
import configparser
from typing import Any, List, Tuple, Union, Dict

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
from typing import Dict
import tempfile
import json
from data_discovery_ai import logger


class Concept:
    def __init__(self, value: str, url: str, vocab_type: str) -> None:
        self.value = value
        self.url = url
        self.vocab_type = vocab_type

    def to_json(self) -> Dict[str, Any]:
        return {
            "vocab_type": self.vocab_type,
            "value": self.value,
            "url": self.url,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Concept):
            return NotImplemented

        return (
            self.value == other.value
            and self.url == other.url
            and self.vocab_type == other.vocab_type
        )

    def __hash__(self):
        return hash((self.value, self.url, self.vocab_type))

    def __str__(self):
        return f"Concept(value={self.value}, url='{self.url}', vocab_type='{self.vocab_type}')"


def save_to_file(obj: Any, full_path: str) -> None:
    """
    Saves an object to a file using pickle serialization in the specific file path
    """
    try:
        with open(full_path, "wb") as file:
            pickle.dump(obj, file)
            logger.info(f"Saved to {full_path}")
    except Exception as e:
        logger.error(e)


def load_from_file(full_path: str) -> Any:
    """
    Loads an object from a file in the input folder using pickle deserialization.
    """
    try:
        with open(full_path, "rb") as file:
            obj = pickle.load(file)
            logger.info(f"Load from {full_path}")
        return obj
    except Exception as e:
        logger.error(e)


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
    preprocessed_data.columns = ["id", "title", "description", "lineage"]

    # add information column, which is the text of title, description and lineage
    preprocessed_data["information"] = (
        preprocessed_data["title"]
        + " [SEP] "
        + preprocessed_data["description"]
        + " [SEP] "
        + preprocessed_data["lineage"]
        + " [SEP]"
    )

    # only focus on onGoing records
    preprocessed_data = preprocessed_data[
        preprocessed_data["_source.summaries.status"] == "onGoing"
    ]
    return preprocessed_data


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
    sampleSet["information"] = (
        sampleSet["title"] + " [SEP] " + sampleSet["description"] + " [SEP]"
    )
    return sampleSet


def prepare_X_Y(
    sampleSet: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Prepares the input feature matrix (X) and target matrix (Y) from the sample set data.
    Input:
        sampleSet: pd.DataFrame. The sample set DataFrame containing feature and target data,
      including an `embedding` column for input features.
    Output:
        X: np.ndarray. A numpy array containing feature variables for items in the sample set.
        Y: np.ndarray. A numpy array containing target variables for items in the sample set.
        Y_df: pd.Dataframe. A DataFrame representation of the target variables.
        labels: Dict. A dict of predefined keyword labels extracted from 'keywords' column. The key is the index of a keyword, and the value is the keyword as a Concept object.
    """
    X = np.array(sampleSet["embedding"].tolist())
    Y_df, labels = prepare_Y_matrix(sampleSet)
    Y = Y_df.to_numpy()
    # TODO: labels set to be List not Dict but identify_rare_labels where labels being consumed expects Dict?
    return X, Y, Y_df, labels


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
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", clean_up_tokenization_spaces=False
    )
    model = BertModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    text_embedding = cls_embedding.squeeze().numpy()
    return text_embedding


def calculate_embedding(ds: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate embeddings for a dataframe, based on the description field, add a embedding column for this dataframe
    Input:
        ds: pd.DataFrame. the dataset that need to be calculated
    Output:
        ds: pd.DataFrame, the dataset with one more embedding column
    """
    tqdm.pandas()
    # add try except
    try:
        ds["embedding"] = ds["information"].progress_apply(
            lambda x: get_description_embedding(x)
        )
    except Exception as e:
        logger.error(e)
    return ds


def prepare_Y_matrix(ds: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the target matrix (Y) by applying multi-label binarization on keywords. This function uses `MultiLabelBinarizer` to transform the `keywords` column of the dataset into a binary matrix format,
    where each unique keyword is represented as a binary feature. The output DataFrame has one column per keyword, with values indicating the presence (1) or absence (0) of each keyword for each record.
    If there are any empty labels, they are removed from the output matrix.

    Input:
        ds: pd.DataFrame. The dataset containing a `keywords` column, where each entry is expected to be a list of keywords.
    Output:
        K: A DataFrame representing the multi-label binarized target matrix, with each column corresponding to a unique keyword.
        keywordMap: A dict represents the predefined keyword set. The key is the index of the keyword, and the value is a keyword defined as a Concept object.
    """
    keywordSet = set()
    for keyword_list in ds["keywords"]:
        for keyword_dict in keyword_list:
            keyword_obj = Concept(
                vocab_type=keyword_dict.get("vocab_type"),
                value=keyword_dict.get("value"),
                url=keyword_dict.get("url"),
            )
            keywordSet.add(keyword_obj)
    keywordMap = {index: keyword for index, keyword in enumerate(keywordSet)}

    ds["keywordsMap"] = ds["keywords"].apply(
        lambda keyword_list: [
            next(
                idx
                for idx, kw in keywordMap.items()
                if kw.vocab_type == d.get("vocab_type")
                and kw.value == d.get("value")
                and kw.url == d.get("url")
            )
            for d in keyword_list
        ]
    )
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(ds["keywordsMap"])
    K = pd.DataFrame(Y, columns=mlb.classes_)

    if "" in K.columns:
        K.drop(columns=[""], inplace=True)
    return K, keywordMap


def keywords_formatter(text: Union[str, List[dict]], vocabs: List[str]) -> List[str]:
    """
    Formats a list of keywords based on specified vocabulary terms. This function processes a list of keywords to identify those matching the specified `vocabs` list. For each matching keyword, it constructs a formatted string of the form `title:id` and removes any duplicates.
    If `text` is a string, it will be evaluated as a list before processing.

    Input:
        text: Union[str, List[dict]. The input keywords, expected to be a list of dictionaries, can be passed as a string representation of the list.
        vocabs: List[str]. A list of vocabulary names to match against keyword titles.
    Output:
        A list of formatted keywords, with duplicates removed, in the form `title;id`.
    """
    if type(text) is list:
        keywords = text
    else:
        keywords = ast.literal_eval(text)
    k_list = []
    for keyword in keywords:
        if keyword.get("concepts") is not None:
            for concept in keyword.get("concepts"):
                if keyword.get("title") in vocabs and concept.get("id") != "":
                    # check if the url is valid: start with http or https
                    try:
                        concept_url = concept.get("url")
                        if re.match(r"^https?://", concept_url):
                            conceptObj = Concept(
                                value=concept.get("id").lower(),
                                url=concept_url,
                                vocab_type=keyword.get("title"),
                            )
                            k_list.append(conceptObj.to_json())
                    except Exception as e:
                        logger.error(e)
                    
    return list(k_list)


def prepare_train_test(
    X: np.ndarray, Y: np.ndarray, params: configparser.ConfigParser
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the training and testing datasets using multi-label stratified splitting.
    This function splits the feature matrix X and target matrix Y into training and testing sets based on parameters for multi-label stratified shuffling. It logger.infos dataset information and returns the dimensions, number of labels, and split data for training and testing.
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

    n_splits = params.getint("preprocessor", "n_splits")
    test_size = params.getfloat("preprocessor", "test_size")
    train_test_random_state = params.getint("preprocessor", "train_test_random_state")
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


def customized_resample(X_train, Y_train, rare_class):
    """
    Customised resampling strategy: given a list of rare labels, i.e., the labels appears under a certain times in alll records, duplicate the records which has these labels with a num_copies.
    Input:
        X_train: np.ndarray. The feature matrix X to be augmented.
        Y_train: np.ndarray. The target matrix Y to be augmented.
        rare_class: List[int]. The index of rare labels.
    Output:
        X_augmented, Y_augmented: Tuple[np.ndarray, np.ndarray]: The augmented training feature matrix and target matrix.
    """
    X_augmented = X_train.copy()
    Y_augmented = Y_train.copy()
    num_copies = 10
    for label_idx in rare_class:
        sample_idx = np.where(Y_train[:, label_idx] == 1)[0]

        if len(sample_idx) == 1:
            sample_to_duplicate = sample_idx[0]
            for _ in range(num_copies):
                X_augmented = np.vstack([X_augmented, X_train[sample_to_duplicate]])
                Y_augmented = np.vstack([Y_augmented, Y_train[sample_to_duplicate]])
    return X_augmented, Y_augmented


def resampling(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    strategy: str,
    rare_keyword_index: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples the training data using the specified strategy to address class imbalance.

    This function applies different resampling strategies to balance the dataset based on the specified `strategy`. For a "custom" strategy, it calls `customized_resample`, which duplicates samples of rare classes.
    Other supported strategies include:
        - "ROS" for random oversampling
        - "RUS" for random undersampling
        - "SMOTE" for synthetic minority oversampling technique

    Input:
        X_train: np.ndarray. The training feature X matrix.
        Y_train: np.ndarray. The training traget Y matrix.
        strategy: str. Resampling strategy to apply ("custom", "ROS", "RUS", or "SMOTE").
        rare_keyword_index: List[int]. A list of indices representing rare class labels for custom resampling.
    Output:
        X_train_resampled, Y_train_resampled: Tuple[np.ndarray, np.ndarray]. The resampled training feature matrix X_train_resampled and target matrix Y_train_resampled.
    """
    Y_train_combined = np.array(["".join(row.astype(str)) for row in Y_train])
    if strategy == "custom":
        X_train_resampled, Y_train_resampled = customized_resample(
            X_train, Y_train, rare_keyword_index
        )
    else:
        if strategy == "ROS":
            resampler = RandomOverSampler(sampling_strategy="auto", random_state=32)
        elif strategy == "RUS":
            resampler = RandomUnderSampler(sampling_strategy="auto", random_state=32)
        elif strategy == "SMOTE":
            resampler = SMOTE(k_neighbors=1, random_state=42)

        X_train_resampled, Y_combined_resampled = resampler.fit_resample(
            X_train, Y_train_combined
        )
        Y_train_resampled = np.array(
            [list(map(int, list(row))) for row in Y_combined_resampled]
        )

    logger.info(" ======== After Resampling ========")
    logger.info(f"Total samples: {len(X_train_resampled)}")
    logger.info(f"Dimension: {X_train_resampled.shape[1]}")
    logger.info(f"No. of labels: {Y_train_resampled.shape[1]}")
    logger.info(f"X resampled set size: {X_train_resampled.shape[0]}")
    logger.info(f"Y resampled set size: {Y_train_resampled.shape[0]}")
    return X_train_resampled, Y_train_resampled
