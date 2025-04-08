from elasticsearch import (
    Elasticsearch,
)
import configparser
import logging
import pandas as pd
from tqdm import tqdm
import time

from data_discovery_ai.common.constants import BATCH_SIZE, SLEEP_TIME, ES_INDEX_NAME
from data_discovery_ai import logger


def connect_es(config: configparser.ConfigParser) -> Elasticsearch:
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
    end_point = config["elasticsearch"]["end_point"]
    api_key = config["elasticsearch"]["api_key"]
    try:
        client = Elasticsearch(end_point, api_key=api_key)
        logging.info("Connected to ElasticSearch")
        return client
    except Exception as e:
        logger.error(f"Elasticsearch connection failed: {e}")


def search_es(
    client: Elasticsearch,
    index: str = ES_INDEX_NAME,
    batch_size: int = BATCH_SIZE,
    sleep_time: int = SLEEP_TIME,
):
    """
    Search elasticsearch index, convert the json format to dataframe, save the dataframe to a pickle file
    Input:
        client: Elasticsearch. The initialised Elasticsearch client instance
        index: str. The index name in Elasticsearch. Default as "es-indexer-staging"
        batch_size: int. The number of documents searches at one query. Please not set a very large number in case to search too deep.
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
        for round in tqdm(range(1, rounds), desc="searching elasticsearch"):
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

                round += 1
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(e)

        # close pit
        try:
            client.close_point_in_time(
                id=current_pit,
            )
        except Exception as e:
            logger.error(f"Failed to close point-in-time: {e}")

        return pd.concat(dataframes, ignore_index=True)

    except Exception as e:
        logger.error(f"Elasticsearch Search Faild: {e}")
