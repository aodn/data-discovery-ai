from elasticsearch import Elasticsearch
import configparser
import json
import logging
import pandas as pd
from tqdm import tqdm
import time
from data_discovery_ai.utils.preprocessor import save_to_file


CONFIG_PATH = "./esManager.config"

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


def connect_es(config_path: str) -> Elasticsearch:

    config = configparser.ConfigParser()
    config.read(config_path)

    end_point = config["elasticsearch"]["end_point"]
    api_key = config["elasticsearch"]["api_key"]

    client = Elasticsearch(end_point, api_key=api_key)
    logging.info("Connected to ElasticSearch")
    return client


"""
    Search elasticsearch index, convert the json format to dataframe, save the dataframe to a pickle file 
    Input:
        client: Elasticsearch. The initialised Elasticsearch client instance
"""


def search_es(client: Elasticsearch):
    index = "es-indexer-staging"
    start_index = 0
    dataframes = []
    batch_size = 100
    resp = client.search(index=index, body={"query": {"match_all": {}}})
    total_number = resp["hits"]["total"]["value"]

    rounds = (total_number + batch_size - 1) // batch_size

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
        rount += 1
        time.sleep(1)

    raw_data = pd.concat(dataframes, ignore_index=True)

    save_to_file(raw_data, "./input/es-indexer-staging.pkl")
    logging.info("Raw data saved to ./input/es-indexer-staging.pkl")

    # TODO: upload raw data to S3
