import ast
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import tempfile

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from data_discovery_ai import logger
import data_discovery_ai.utils.es_connector as es_connector
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.agent_tools import get_text_embedding


class BasePreprocessor:
    def __init__(self):
        self.config = ConfigUtil()
        self.temp_dir = tempfile.mkdtemp()
        self.watched_columns = []
        self.replaced_column_names = []
        self.require_embedding = []
        self.data = None

    def fetch_raw_data(self) -> pd.DataFrame | None:
        es_config = self.config.get_es_config()
        client = es_connector.connect_es()
        if client:
            raw_data = es_connector.search_es(
                client=client,
                index=es_config["es_index_name"],
                batch_size=es_config["batch_size"],
                sleep_time=es_config["sleep_time"],
            )
            return raw_data
        else:
            return None

    def post_filter_hook(self, df: pd.DataFrame) -> pd.DataFrame | None:
        return df

    def filter_raw_data(self, raw_data):
        filtered_data = raw_data[self.watched_columns]
        if self.replaced_column_names:
            filtered_data.columns = self.replaced_column_names
        return self.post_filter_hook(filtered_data)

    def calculate_embedding(
        self, ds: pd.DataFrame, seperator: str
    ) -> pd.DataFrame | None:
        """
        Calculate embeddings for a dataframe, based on the description field, add a embedding column for this dataframe
        Input:
            ds: pd.DataFrame. the dataset that need to be calculated
        Output:
            ds: pd.DataFrame, the dataset with one more embedding column or None if calculation failed
        """
        tqdm.pandas()
        text_columns = self.require_embedding
        # try:
        ds[text_columns] = ds[text_columns].fillna("").astype(str)
        ds["combined_text"] = ds[text_columns].agg(seperator.join, axis=1)

        ds["embedding"] = ds["combined_text"].progress_apply(
            lambda x: get_text_embedding(x)
        )
        return ds
        # except Exception as e:
        #     logger.error(f"Failed to calculate embeddings: {e}")
        #     return None

    def set_preprocessed_data(self, df: pd.DataFrame) -> None:
        pass


@dataclass
class KMData:
    X: np.ndarray
    Y: np.ndarray
    Y_df: pd.DataFrame
    labels: Dict[int, Any]


@dataclass
class KMTrainTestData:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    label_weight_dict: Dict[int, float]
    dimension: int
    n_labels: int


def resampling(
    X_train: np.ndarray, Y_train: np.ndarray, strategy: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples the training data using the specified strategy to address class imbalance.

    This function applies different resampling strategies to balance the dataset based on the specified `strategy`. Supported strategies include:
        - "ROS" for random oversampling
        - "RUS" for random undersampling
        - "SMOTE" for synthetic minority oversampling technique

    Input:
        strategy: str. Resampling strategy to apply ("ROS", "RUS", or "SMOTE").
    Output:
        X_train_resampled, Y_train_resampled: Tuple[np.ndarray, np.ndarray]. The resampled training feature matrix X_train_resampled and target matrix Y_train_resampled.
    """
    Y_train_combined = np.array(["".join(row.astype(str)) for row in Y_train])
    resampler = None
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


class KeywordPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()
        self.trainer_config = self.config.get_keyword_trainer_config()

        self.watched_columns = [
            "_id",
            "_source.title",
            "_source.description",
            "_source.themes",
        ]
        self.replaced_column_names = ["id", "title", "abstract", "keywords"]
        self.require_embedding = ["title", "abstract"]
        self.rare_labels = []
        self.data = None
        self.train_test_data = None

    def set_preprocessed_data(self, df: pd.DataFrame) -> None:
        Y_df, labels = prepare_Y_matrix(df)
        Y = Y_df.to_numpy()
        self.data = KMData(
            X=np.array(df["embedding"].tolist()), Y=Y, Y_df=Y_df, labels=labels
        )

    def set_rare_labels(self):
        rare_label_threshold = self.trainer_config["rare_label_threshold"]
        label_df = self.data.Y_df.copy()
        label_distribution = label_df.sum().sort_values().to_frame(name="count")
        rare_labels = label_distribution[
            label_distribution["count"] <= rare_label_threshold
        ].index.to_list()
        self.rare_labels = rare_labels

    def fetch_raw_data(self) -> pd.DataFrame | None:
        raw_data = super().fetch_raw_data()
        return raw_data

    def post_filter_hook(self, df: pd.DataFrame) -> pd.DataFrame | None:
        watched_vocabs = self.trainer_config["vocabs"]
        # only keep rows with vocabs in selected vocabs
        watched_df = df[
            df["keywords"].apply(
                lambda terms: any(
                    any(vocab in k["title"] for vocab in watched_vocabs) for k in terms
                )
            )
        ]
        # format keyword values into Concept object
        watched_df["keywords"] = watched_df["keywords"].apply(
            lambda x: keywords_formatter(x, watched_vocabs)
        )

        # clean rows with empty keywords
        return watched_df[watched_df["keywords"].apply(lambda x: x != [])]

    def customized_resample(self):
        """
        Customised resampling strategy: given a list of rare labels, i.e., the labels appears under a certain times in alll records, duplicate the records which has these labels with a num_copies.
        Input:
            X_train: np.ndarray. The feature matrix X to be augmented.
            Y_train: np.ndarray. The target matrix Y to be augmented.
            rare_class: List[int]. The index of rare labels.
        Output:
            X_augmented, Y_augmented: Tuple[np.ndarray, np.ndarray]: The augmented training feature matrix and target matrix.
        """
        X_augmented = self.data.X.copy()
        Y_augmented = self.data.Y.copy()
        num_copies = self.trainer_config["rare_label_threshold"]

        for label_idx in self.rare_labels:
            positive_idx = np.where(self.data.Y[:, label_idx] == 1)[0]
            count_pos = len(positive_idx)
            if 0 < count_pos <= num_copies:
                need = num_copies - count_pos

                dup_idx = np.random.choice(positive_idx, size=need, replace=True)
                for i in dup_idx:
                    X_augmented = np.vstack([X_augmented, self.data.X[i]])
                    Y_augmented = np.vstack([Y_augmented, self.data.Y[i]])

        return X_augmented, Y_augmented

    def prepare_train_test_set(self, raw_data: pd.DataFrame) -> None:
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
        # set preprocessed data
        self.set_preprocessed_data(raw_data)

        # set rare labels
        self.set_rare_labels()

        # customised resampling
        X_augmented, Y_augmented = self.customized_resample()

        n_labels = Y_augmented.shape[1]
        dim = X_augmented.shape[1]

        shuffle_split = MultilabelStratifiedShuffleSplit(
            n_splits=self.trainer_config["n_splits"],
            test_size=self.trainer_config["test_size"],
            random_state=42,
        )
        X_train, Y_train, X_test, Y_test = None, None, None, None
        for train_index, test_index in shuffle_split.split(X_augmented, Y_augmented):
            X_train, X_test = X_augmented[train_index], X_augmented[test_index]
            Y_train, Y_test = Y_augmented[train_index], Y_augmented[test_index]

        # X_train_resampled, Y_train_resampled = resampling(X_train, Y_train, "ROS")

        label_weight_dict = get_class_weights(Y_train)

        self.train_test_data = KMTrainTestData(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            dimension=dim,
            n_labels=n_labels,
            label_weight_dict=label_weight_dict,
        )


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


def prepare_Y_matrix(ds: pd.DataFrame) -> tuple[DataFrame, dict[int, Any]]:
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
        lambda kl: [
            next(
                idx
                for idx, kw in keywordMap.items()
                if kw.vocab_type == d.get("vocab_type")
                and kw.value == d.get("value")
                and kw.url == d.get("url")
            )
            for d in kl
        ]
    )
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(ds["keywordsMap"])
    K = pd.DataFrame(Y, columns=mlb.classes_)

    if "" in K.columns:
        K.drop(columns=[""], inplace=True)
    return K, keywordMap


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


class DeliveryPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()
        self.trainer_config = self.config.get_delivery_trainer_config()

        self.watched_columns = [
            "_id",
            "_source.title",
            "_source.description",
            "_source.summaries.statement",
            "_source.summaries.status",
        ]
        self.replaced_column_names = ["id", "title", "abstract", "lineage", "status"]
        self.require_embedding = ["title", "abstract", "lineage"]
        self.data = None
        self.train_test_data = None

    def post_filter_hook(self, df: pd.DataFrame) -> pd.DataFrame | None:
        filtered_df = df[df["status"] == "onGoing"]

        # make a copy of the input data to ensure the original data is not modified
        temp = filtered_df.copy()

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
        data_with_mode = filtered_df.join(real_time_delayed_data["mode"])

        label_map = {"Real-Time": 0, "Delayed": 1, np.nan: -1}
        data_with_mode["mode"] = data_with_mode["mode"].map(label_map)
        return filtered_df

    def fetch_raw_data(self) -> pd.DataFrame | None:
        return super().fetch_raw_data()
