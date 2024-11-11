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
    start_index = 0
    dataframes = []
    batch_size = 100
    resp = client.search(index=index, body={"query": {"match_all": {}}})
    total_number = resp["hits"]["total"]["value"]

    rounds = (total_number + batch_size - 1) // batch_size

    # TODO: refactor the search query method.
    for round in tqdm(range(rounds), desc="searching elasticsearch"):
        search_query = {
            "size": batch_size,
            "from": start_index,
            "query": {"match_all": {}},
        }

        resp = client.search(index=index, body=search_query)

        data = resp["hits"]["hits"]
        df = pd.json_normalize(data)
        dataframes.append(df)

        start_index += 1
        round += 1
        time.sleep(1)

    raw_data = pd.concat(dataframes, ignore_index=True)

    return raw_data
