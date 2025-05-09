import ast
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import tempfile

from tqdm import tqdm

from data_discovery_ai import logger
import data_discovery_ai.utils.es_connector as es_connector
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.agent_tools import get_text_embedding


def filter_with_column_name(
    data: pd.DataFrame,
    selected_columns: List[str],
    replace_column_name: Optional[List[str]] = None,
):
    filtered_data = data[selected_columns]
    if replace_column_name:
        filtered_data.columns = replace_column_name
    return filtered_data


class BasePreprocessor:
    def __init__(self):
        self.config = ConfigUtil()
        self.temp_dir = tempfile.mkdtemp()
        self.watched_columns = []
        self.replaced_column_names = []
        self.require_embedding = []

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
        filtered_data = filter_with_column_name(
            data=raw_data,
            selected_columns=self.watched_columns,
            replace_column_name=self.replaced_column_names,
        )
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
        try:
            ds[text_columns] = ds[text_columns].fillna("").astype(str)
            ds["combined_text"] = ds[text_columns].agg(seperator.join, axis=1)
            ds["embedding"] = ds["combined_text"].progress_apply(
                lambda x: get_text_embedding(x)
            )
            return ds
        except Exception as e:
            logger.error(f"Failed to calculate embeddings: {e}")
            return None


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
        self.required_embedding = ["title", "abstract", "lineage"]

    def post_filter_hook(self, df: pd.DataFrame) -> pd.DataFrame | None:
        filtered_df = df[df["status"] == "onGoing"]
        return filtered_df

    def fetch_raw_data(self) -> pd.DataFrame | None:
        return super().fetch_raw_data()
