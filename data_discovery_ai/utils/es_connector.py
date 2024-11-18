from elasticsearch import (
    Elasticsearch,
)  # TODO: please use poetry add command to install any new libraries
import configparser
import logging
import pandas as pd
from tqdm import tqdm
import time


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


def connect_es(config: configparser.ConfigParser) -> Elasticsearch:
    end_point = config["elasticsearch"]["end_point"]
    api_key = config["elasticsearch"]["api_key"]

    client = Elasticsearch(end_point, api_key=api_key)
    logging.info("Connected to ElasticSearch")
    return client


"""
    Search elasticsearch index, convert the json format to dataframe, save the dataframe to a pickle file
    Input:
        client: Elasticsearch. The initialised Elasticsearch client instance
    Output:
        raw_data: pd.DataFrame. The fetched raw data in a tabular format.
"""


def search_es(client: Elasticsearch):
    index = "es-indexer-staging"
    dataframes = []
    batch_size = 100

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
        "size": 100,
        "query": {"match_all": {}},
        "pit": {"id": first_pit, "keep_alive": "1m"},
        "sort": [{"id.keyword": "asc"}],
    }
    first_query_resp = client.search(body=first_query_body)

    data = first_query_resp["hits"]["hits"]
    # set search after value
    current_last_result = data[-1]["sort"]

    # save the first search result
    df = pd.json_normalize(data)
    dataframes.append(df)

    # set current pit as the first one
    current_pit = first_query_resp["pit_id"]

    # conduct further search
    for round in tqdm(range(1, rounds), desc="searching elasticsearch"):
        qurry_body = {
            "size": 100,
            "query": {"match_all": {}},
            "pit": {"id": current_pit, "keep_alive": "1m"},
            "sort": [{"id.keyword": "asc"}],
            "search_after": current_last_result,
            "track_total_hits": False,
        }
        query_resp = client.search(body=qurry_body)

        data = query_resp["hits"]["hits"]
        # set search after value
        current_last_result = data[-1]["sort"]

        # save the first search result
        df = pd.json_normalize(data)
        dataframes.append(df)

        # set current pit as the first one
        current_pit = query_resp["pit_id"]

        round += 1
        time.sleep(10)

    # close pit
    resp = client.close_point_in_time(
        id=current_pit,
    )
    print(resp)

    raw_data = pd.concat(dataframes, ignore_index=True)
    return raw_data
