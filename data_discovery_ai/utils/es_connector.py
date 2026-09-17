from elastic_transport import ObjectApiResponse
from elasticsearch import (
    Elasticsearch,
    ApiError,
    BadRequestError,
    NotFoundError,
    ConnectionError as ESConnectionError,
    ConnectionTimeout,
)
import hashlib
import os
from dotenv import load_dotenv
import json
from typing import Tuple, Dict, Any
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    stop_before_delay,
    wait_exponential_jitter,
)

from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.config.constants import RECORDS_ENHANCED_SCHEMA
from data_discovery_ai.utils.retry_template import RETRYABLE_HTTP_STATUS, log_retry

logger = structlog.get_logger(__name__)

ES_STARTUP_REQUEST_TIMEOUT = 5.0
ES_REQUEST_TIMEOUT = 5.0


def _es_status_retryable(exc: BaseException) -> bool:
    # 429 and 5xx are raised as plain ApiError, so check the status code
    return isinstance(exc, ApiError) and exc.meta.status in RETRYABLE_HTTP_STATUS


_es_retryable = (
    # Network-level failures: node unreachable, connection reset, request timeout
    retry_if_exception_type((ESConnectionError, ConnectionTimeout))
    | retry_if_exception(_es_status_retryable)
)

# for startup connection, e.g., connecting ES, creating index
es_startup_retry = retry(
    retry=_es_retryable,
    # max_attempts includes the first call; stop_before_delay caps total elapsed time
    stop=stop_after_attempt(6) | stop_before_delay(60),
    wait=wait_exponential_jitter(initial=1, max=10),
    before_sleep=log_retry,
    # Raise the original exception instead of tenacity.RetryError
    reraise=True,
)

# for querying ES, e.g., querying, deleting, storing data
es_request_retry = retry(
    retry=_es_retryable,
    stop=stop_after_attempt(3) | stop_before_delay(15),
    wait=wait_exponential_jitter(initial=1, max=4),
    before_sleep=log_retry,
    reraise=True,
)


def _error_type(exc: ApiError) -> str | None:
    """Extract the ES error type, e.g. 'resource_already_exists_exception'."""
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error")
    return error.get("type") if isinstance(error, dict) else None


@es_startup_retry
def _index_exists(client: Elasticsearch, index: str) -> bool:
    startup_client = client.options(request_timeout=ES_STARTUP_REQUEST_TIMEOUT)
    return bool(startup_client.indices.exists(index=index))


@es_startup_retry
def _get_stored_hash(client: Elasticsearch, index_name: str) -> str | None:
    try:
        startup_client = client.options(request_timeout=ES_STARTUP_REQUEST_TIMEOUT)
        mappings = startup_client.indices.get_mapping(index=index_name)
    except NotFoundError:
        # Only a real 404 means "no hash"; other errors are retried and raised
        return None
    return mappings[index_name]["mappings"].get("_meta", {}).get("mapping_hash")


@es_startup_retry
def _delete_index(client: Elasticsearch, index: str) -> None:
    try:
        startup_client = client.options(request_timeout=ES_STARTUP_REQUEST_TIMEOUT)
        startup_client.indices.delete(index=index)
    except NotFoundError:
        # A previous attempt may have succeeded but its response was lost
        logger.info(f"Elasticsearch index '{index}' already absent, skip deleting.")


@es_startup_retry
def _create_index(client: Elasticsearch, index: str, mapping: dict) -> None:
    try:
        startup_client = client.options(request_timeout=ES_STARTUP_REQUEST_TIMEOUT)
        startup_client.indices.create(index=index, body=mapping)
    except BadRequestError as e:
        # A previous attempt may have created the index before timing out
        if _error_type(e) == "resource_already_exists_exception":
            logger.info(f"Elasticsearch index '{index}' already exists, skip creating.")
            return
        raise


@es_request_retry
def _index_document(
    client: Elasticsearch, index: str, doc_id: str, data: Dict[Any, Any]
) -> None:
    # Indexing with an explicit id is idempotent, so retrying is safe
    request_client = client.options(request_timeout=ES_REQUEST_TIMEOUT)
    request_client.index(index=index, document=data, id=doc_id)


def _delete_document(client: Elasticsearch, index: str, uuid: str) -> str:
    request_client = client.options(request_timeout=ES_REQUEST_TIMEOUT)
    attempted = False

    @es_request_retry
    def _attempt() -> str:
        nonlocal attempted
        try:
            return request_client.delete(index=index, id=uuid).get("result")
        except NotFoundError:
            # A 404 after an ambiguous failure means the earlier attempt deleted it.
            return "deleted" if attempted else "not_found"
        except Exception:
            attempted = True
            raise

    return _attempt()


@es_request_retry
def search_es_documents(
    client: Elasticsearch, index: str, query: dict
) -> ObjectApiResponse[Any]:
    """Search Elasticsearch using the request-path retry budget."""
    request_client = client.options(request_timeout=ES_REQUEST_TIMEOUT)
    return request_client.search(index=index, body=query)


def connect_es() -> Elasticsearch | None:
    """
    Function to connect the ElasticSearch
    Input:
        config_path: str. The config file path to store the end_point and api_key information. Formatted as:
                    [elasticsearch]
                        end_point="elasticsearch_end_point"
                        api_key="elasticsearch_api_key"
    Output:
        client:Elasticsearch. An initialised Elasticsearch client instance. Or None if initialisation failed.
    """
    load_dotenv()

    end_point = os.getenv("ES_ENDPOINT")
    api_key = os.getenv("ES_API_KEY")
    try:
        # Connectivity is verified by the first retried index call in create_es_index.
        client = Elasticsearch(end_point, api_key=api_key, max_retries=0)
        logger.info("Elasticsearch client initialised")
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
    """Retrieve the mapping hash stored in the index metadata with retry policy."""
    return _get_stored_hash(client, index_name)


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

    try:
        exists = _index_exists(client, index)
        stored_hash = get_stored_hash(client, index) if exists else None
    except Exception as e:
        logger.error(f"Failed to inspect Elasticsearch index '{index}': {e}")
        return None, None

    if exists:
        # no schema change, keep using the current index
        if stored_hash == current_hash:
            # Mapping is unchanged, no action needed
            logger.info(f"Elasticsearch index '{index}' mapping unchanged, reusing.")
            return client, index

        # Mapping has changed, rebuild the index
        logger.warning(f"Mapping changed for index '{index}', rebuilding...")
        try:
            _delete_index(client, index)
            logger.info(f"Elasticsearch index '{index}' deleted.")
        except Exception as e:
            logger.error(f"Failed to delete Elasticsearch index '{index}': {e}")
            return None, None

    # Creating index with configured schema. Inject hash into mapping settings before creating
    mapping.setdefault("mappings", {})
    mapping["mappings"].setdefault("_meta", {})
    mapping["mappings"]["_meta"]["mapping_hash"] = current_hash

    try:
        _create_index(client, index, mapping)
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

    _index_document(client, index, doc_id, data)
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
        result = _delete_document(client, index, uuid)
    except Exception as e:
        logger.error(f"Error deleting document '{uuid}': {e}")
        return False

    if result == "not_found":
        logger.warning(f"Document '{uuid}' not found in index '{index}'.")
    return result == "deleted"
