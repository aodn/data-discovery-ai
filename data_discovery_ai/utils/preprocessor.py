"""
    The module to preprocess the data and prepare datasets for training and evaluating a ML model
"""

import pickle
import pandas as pd
import ast
import numpy as np
import configparser
from typing import Any, List, Tuple, Union, Optional
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
from typing import Dict

from data_discovery_ai.config.constants import RARE_LABEL_THRESHOLD

# TODO: use the below line after fix 'dada_discovery_ai' module not exist issue in notebook: ModuleNotFoundError: No module named 'data_discovery_ai'
# from data_discovery_ai import logger

# TODO: remove this after fix 'dada_discovery_ai' module not exist issue in notebook: ModuleNotFoundError: No module named 'data_discovery_ai'
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# hide warning information from transformers
from transformers import logging as tf_logging

tf_logging.set_verbosity_error()


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


def label_ddm_sample(filtered_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to label the "onGoing" records based on a decision-making tree.
    If the title of a record contains "Real-Time" or its variants (ignore case), the records is labelled as "Real-Time" mode. Similarly, if the title of a record contains "Delayed" or its variants (ignore case), the records is labelled as "Delayed" mode.
    Otherwise, the record remains unlabelled.
    The mode is mapped with numbers: 0 for "Real-Time", 1 for "Delayed", and -1 for unlabelled records.
    Input:
        filtered_data: pd.DataFrame. The preprocessed DataFrame which is expected to have these fields: "id", "title", "abstract", "lineage", "status", "information", "embedding".
    Output:
        sampeSet: pd.DataFrame. The final data set that contains both the labelled and unlabelled records. It is the same as the input with one more "mode" column.
    """
    # make a copy of the input data to ensure the original data is not modified
    temp = filtered_data.copy()

    # find rows with title contains 'real time' and its variants
    # define real time string and variants and ignore case
    real_time_variants = ["real time", "real-time", "realtime"]
    real_time_data = temp[
        temp["title"].str.contains("|".join(real_time_variants), case=False)
    ]
    real_time_data.loc[:, "mode"] = "Real-Time"
    # and also for 'delayed' and its variants
    delayed_variants = ["delayed", "delay", "delaying"]
    delayed_data = temp[
        temp["title"].str.contains("|".join(delayed_variants), case=False)
    ]
    delayed_data.loc[:, "mode"] = "Delayed"

    real_time_delayed_data = pd.concat([real_time_data, delayed_data])
    data_with_mode = filtered_data.join(real_time_delayed_data["mode"])

    label_map = {"Real-Time": 0, "Delayed": 1, np.nan: -1}
    data_with_mode["mode"] = data_with_mode["mode"].map(label_map)

    return data_with_mode


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


def prepare_X_Y(
    sampleSet: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict]:
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
        try:
            if keyword.get("concepts") is not None:
                for concept in keyword.get("concepts"):
                    if keyword.get("title") in vocabs and concept.get("id") != "":
                        # check if the url is valid: start with http and not None or empty
                        if concept.get("url") is not None and concept.get("url") != "":
                            concept_url = concept.get("url")
                            if concept_url.startswith("http"):
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

    # set the number of copies as the same of the rare label threshold so that no need to manually adjust this value
    num_copies = RARE_LABEL_THRESHOLD
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
    rare_keyword_index: Optional[List[int]],
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
        rare_keyword_index: List[int] or None. List[int] as a list of indices representing rare class labels for custom resampling or None for ROS, RUS, and SMOTE.
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
