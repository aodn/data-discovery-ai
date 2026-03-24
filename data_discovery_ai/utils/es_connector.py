from elasticsearch import Elasticsearch
import logging
import hashlib
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


def get_mapping_hash(mapping: dict) -> str:
    """Generate a stable MD5 hash from the mapping dict."""
    # Sort keys to ensure consistent ordering before hashing
    mapping_str = json.dumps(mapping, sort_keys=True)
    return hashlib.md5(mapping_str.encode()).hexdigest()


def get_stored_hash(client: Elasticsearch, index_name: str) -> str | None:
    """Retrieve the mapping hash stored in the index metadata."""
    try:
        mappings = client.indices.get_mapping(index=index_name)
        return mappings[index_name]["mappings"].get("_meta", {}).get("mapping_hash")
    except Exception:
        # Index doesn't exist yet
        return None


def create_es_index() -> Tuple[None, None] | Tuple[Elasticsearch, str]:
    """
    Create Elasticsearch index to store documents with AI-generated data.
    - If index does not exist: create it
    - If index exists and mapping is unchanged: reuse it
    - If index exists and mapping has changed: delete and recreate
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

    current_hash = get_mapping_hash(mapping)

    if client.indices.exists(index=index):
        stored_hash = get_stored_hash(client, index)

        if stored_hash == current_hash:
            # Mapping is unchanged, no action needed
            logger.info(f"Elasticsearch index '{index}' mapping unchanged, reusing.")
            return client, index

        # Mapping has changed, rebuild the index
        logger.warning(f"Mapping changed for index '{index}', rebuilding...")
        try:
            client.indices.delete(index=index)
            logger.info(f"Elasticsearch index '{index}' deleted.")
        except Exception as e:
            logger.error(f"Failed to delete Elasticsearch index '{index}': {e}")
            return None, None

    # Inject hash into mapping settings before creating
    mapping.setdefault("mappings", {})
    mapping["mappings"].setdefault("_meta", {})
    mapping["mappings"]["_meta"]["mapping_hash"] = current_hash

    try:
        client.indices.create(index=index, body=mapping)
        logger.info(
            f"Elasticsearch index '{index}' created with hash {current_hash[:8]}."
        )
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
