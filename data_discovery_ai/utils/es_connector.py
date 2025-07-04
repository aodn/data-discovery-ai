from elasticsearch import Elasticsearch
import logging
import pandas as pd
from pandas import DataFrame
import time
import os
from dotenv import load_dotenv
import json

from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.config.constants import RECORDS_ENHANCED_SCHEMA
from data_discovery_ai import logger


def connect_es() -> Elasticsearch | None:
    """
    Function to connect the ElasticSearch
    Input:
        config_path: str. The config file path to store the end_point and api_key information. Formatted as:
                    [elasticsearch]
                        end_point="elasticsearch_end_point"
                        api_key="elasticsearch_api_key"
    Output:
        client:Elasticsearch. An initialised Elasticsearch client instance.
    """
    load_dotenv()

    end_point = os.getenv("ES_ENDPOINT")
    api_key = os.getenv("ES_API_KEY")
    try:
        client = Elasticsearch(end_point, api_key=api_key)
        logging.info("Connected to ElasticSearch")
        return client
    except Exception as e:
        logger.error(f"Elasticsearch connection failed: {e}")
        return None


def search_es(
    client: Elasticsearch,
    index: str,
    batch_size: int,
    sleep_time: int,
) -> DataFrame | None:
    """
    Search elasticsearch index, convert the json format to dataframe, save the dataframe to a pickle file
    Input:
        client: Elasticsearch. The initialised Elasticsearch client instance
        index: str. The index name in Elasticsearch. Default as "es-indexer-staging"
        batch_size: int. The number of documents searches at one query. Please do not set a very large number in case to search too deep.
        sleep_time: int. The number of seconds of sleep time between each query. This should be set with consideration of 'batch_size'. If 'batch_size' is large, set 'sleep_time' to a large number to ensure each query is finished and does not impact the next query.
    Output:
        raw_data: pd.DataFrame. The fetched raw data in a tabular format.
    """
    dataframes = []

    # get document count
    count_resp = client.count(index=index)
    total_number = count_resp["count"]
    rounds = (total_number + batch_size - 1) // batch_size

    # get the first pit
    first_pit_resp = client.open_point_in_time(
        index=index,
        keep_alive="1m",
    )
    first_pit = first_pit_resp["id"]

    # the first search
    first_query_body = {
        "size": batch_size,
        "query": {"match_all": {}},
        "pit": {"id": first_pit, "keep_alive": "1m"},
        "sort": [{"id.keyword": "asc"}],
    }
    first_query_resp = client.search(body=first_query_body)

    try:
        if "hits" not in first_query_resp or "hits" not in first_query_resp["hits"]:
            raise KeyError("Invalid first query response: missing 'hits.hits'.")

        data = first_query_resp["hits"]["hits"]
        if not data:
            logger.warning("First query response returned no results.")
            return pd.DataFrame()

        # set search after value
        if "sort" not in data[-1]:
            raise KeyError("The last result in 'hits.hits' is missing the 'sort' key.")
        current_last_result = data[-1]["sort"]

        # save the first search result
        df = pd.json_normalize(data)
        dataframes.append(df)

        # set current pit as the first one
        if "pit_id" not in first_query_resp:
            raise KeyError("First query response is missing 'pit_id'.")
        current_pit = first_query_resp["pit_id"]

        # conduct further search
        for r in range(1, rounds):
            query_body = {
                "size": batch_size,
                "query": {"match_all": {}},
                "pit": {"id": current_pit, "keep_alive": "1m"},
                "sort": [{"id.keyword": "asc"}],
                "search_after": current_last_result,
                "track_total_hits": False,
            }
            query_resp = client.search(body=query_body)

            try:
                if (
                    "hits" not in first_query_resp
                    or "hits" not in first_query_resp["hits"]
                ):
                    raise KeyError("Invalid first query response: missing 'hits.hits'.")
                data = query_resp["hits"]["hits"]
                if not data:
                    logger.info("No more results returned. Ending search.")
                    break

                # set search after value
                if "sort" not in data[-1]:
                    raise KeyError(
                        "The last result in 'hits.hits' is missing the 'sort' key."
                    )

                current_last_result = data[-1]["sort"]

                # save the current search result
                df = pd.json_normalize(data)
                dataframes.append(df)

                # set current pit
                if "pit_id" not in first_query_resp:
                    raise KeyError("First query response is missing 'pit_id'.")
                current_pit = first_query_resp["pit_id"]

                r += 1
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(e)

        # close pit
        try:
            logger.info(f"Total results: {len(dataframes)}")
            client.close_point_in_time(
                id=current_pit,
            )
        except Exception as e:
            logger.error(f"Failed to close point-in-time: {e}")

        return pd.concat(dataframes, ignore_index=True)

    except Exception as e:
        logger.error(f"Elasticsearch Search Failed: {e}")
        return None


def create_es_index():
    """
    Create Elasticsearch index to store documents with AI-generated data. No action applied if the index already exists.
    """
    config = ConfigUtil()
    es_config = config.get_es_config()
    index = es_config.es_ai_index_name
    client = connect_es()

    if client is None:
        return

    schema_path = config.base_dir / "config" / RECORDS_ENHANCED_SCHEMA
    if not os.path.exists(schema_path):
        logger.error(f"Schema file '{schema_path}' not found.")
        raise FileNotFoundError(f"Schema file '{schema_path}' not found.")

    with open(schema_path, "r") as f:
        mapping = json.load(f)

    if client.indices.exists(index=index):
        logger.warning(f"Elasticsearch index '{index}' already exists.")
        return

    try:
        client.indices.create(index=index, body=mapping)
        logger.info(f"Elasticsearch index '{index}' created.")
    except Exception as e:
        logger.error(f"Failed to create Elasticsearch index '{index}': {e}")
        return
