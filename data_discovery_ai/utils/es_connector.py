from elasticsearch import Elasticsearch
import logging
import pandas as pd
from pandas import DataFrame
import time
import os
from dotenv import load_dotenv
import json
from typing import Tuple, Dict, Any
import structlog

from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.config.constants import RECORDS_ENHANCED_SCHEMA

logger = structlog.get_logger(__name__)


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


def create_es_index() -> Tuple[None, None] | Tuple[Elasticsearch, str]:
    """
    Create Elasticsearch index to store documents with AI-generated data. No action applied if the index already exists.
    Output:
        Tuple[Elasticsearch, str]: Elasticsearch client and index if connected successfully. None otherwise.
    """
    config = ConfigUtil.get_config()
    es_config = config.get_es_config()
    index = es_config.es_ai_index_name
    client = connect_es()

    if client is None:
        return None, None

    schema_path = config.base_dir / "config" / RECORDS_ENHANCED_SCHEMA
    if not os.path.exists(schema_path):
        logger.error(f"Schema file '{schema_path}' not found.")
        raise FileNotFoundError(f"Schema file '{schema_path}' not found.")

    with open(schema_path, "r") as f:
        mapping = json.load(f)

    if client.indices.exists(index=index):
        logger.warning(f"Elasticsearch index '{index}' already exists.")
        return client, index

    try:
        client.indices.create(index=index, body=mapping)
        logger.info(f"Elasticsearch index '{index}' created.")
        return client, index
    except Exception as e:
        logger.error(f"Failed to create Elasticsearch index '{index}': {e}")
        return None, None


def store_ai_generated_data(
    data: Dict[Any, Any], client: Elasticsearch, index: str
) -> None:
    """
    Store a document into Elasticsearch with specified index.
    Input:
        data: data to store.
        client: Elasticsearch client.
        index: Elasticsearch index.
    """
    if client is None:
        logger.error(f"Elasticsearch index '{index}' connected failed.")
        return
    doc_id = data["id"]

    client.index(index=index, document=data, id=doc_id)
    logger.info(
        f"Elasticsearch document with uuid '{doc_id}' stored in index '{index}'."
    )


def delete_es_document(uuid: str, client: Elasticsearch, index: str) -> bool:
    """
    Delete a document from an Elasticsearch index with the document id.
    Input:
        uuid: str, the document id.
        client: Elasticsearch client.
        index: str, the index name.
    Output:
        bool, True if the document was deleted. False otherwise.
    """
    if client is None:
        logger.error(f"Failed to connect to Elasticsearch index '{index}'.")
        return False
    try:
        resp = client.delete(index=index, id=uuid)
        return True if resp.get("result") == "deleted" else False
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return False
