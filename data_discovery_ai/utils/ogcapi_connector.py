import time
from typing import Optional, List, Any
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import pandas as pd

import structlog
import requests

from data_discovery_ai.config.config import ConfigUtil

logger = structlog.get_logger(__name__)


class OGCAPIConnector:
    SEARCH_AFTER_PARAM = "search_after"
    FILTER_SEPARATOR = "||"
    FILTER_PREFIX = "='"
    FILTER_SUFFIX = "'"
    QUERY_CONNECTOR = "+AND+"

    def __init__(self, config: Optional["ConfigUtil"] = None):
        if config is None:
            config = ConfigUtil.get_config()

        # get OGCAPI Config
        ogcapi_config = config.get_ogcapi_config()
        self.host = ogcapi_config.host
        self.endpoint = ogcapi_config.endpoint
        self.page_size = ogcapi_config.page_size
        self.sleep_time = ogcapi_config.sleep_time
        self.query = ogcapi_config.query
        self.max_retries = ogcapi_config.max_retries
        self.timeout = ogcapi_config.timeout

        if not self.host:
            raise ValueError("OGCAPI host is not available")
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """
        Create request session for OGCAPI query for retry purpose.
        :return requests.Session
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def build_search_after_filter(self, search_after: List[Any]) -> str:
        """
        Build the encoded search after filter. The search_after value comes from the response.
        :param search_after:
        :return: encoded search after filter string
        """
        filter_value = self.FILTER_SEPARATOR.join(str(item) for item in search_after)
        filter_str = f"{self.SEARCH_AFTER_PARAM}{self.FILTER_PREFIX}{filter_value}{self.FILTER_SUFFIX}"
        encoded_filter = quote(filter_str, safe="")
        encoded_search_after_filter = self.QUERY_CONNECTOR + encoded_filter
        return encoded_search_after_filter

    def build_ogcapi_query_url(self) -> str:
        """
        Build the OGCAPI query url with configured host, endpoint, and page_size. For example:
         http://localhost:8080/api/v1/ogc/collections?properties=id,title,description,statement,themes,status&filter=page_size%3D3
        :return: OGCAPI query url
        """
        base_url = self.host.rstrip("/")
        endpoint = self.endpoint.lstrip("/")
        page_size_str = str(self.page_size)

        ogcapi_query_url = f"{base_url}/{endpoint}{self.query}{page_size_str}"

        logger.debug(f"OGCAPI query url: {ogcapi_query_url}")
        return ogcapi_query_url

    def _parse_fetched_collection_data(self, resp_dict: dict) -> List[dict]:
        """
        Convert response from OGCAPI query to formatted dict as a collection list. Only keep required keys for training ML models.
        Required keys: id, title, description, statement, themes, status.
        :param resp_dict: the response from OGCAPI query
        :return: the parsed collection list
        """
        parsed_collection_data = []
        for collection in resp_dict.get("collections", []):
            props = collection.get("properties", {})
            parsed_collection_data.append(
                {
                    "id": collection.get("id"),
                    "title": collection.get("title"),
                    "description": collection.get("description"),
                    "statement": props.get("statement", None),
                    "themes": props.get("themes", None),
                    "status": props.get("status", None),
                }
            )
        return parsed_collection_data

    def get_all_collections(self):
        """
        Retrieve all collections from the OGCAPI endpoint using search-after. This method performs repeated GET requests
        with sleep time to avoid overwhelming the upstream API, and automatically merges all pages into a single DataFrame.
        :return: pd.DataFrame
        A DataFrame containing all collection records. Columns include:
        - id
        - title
        - description
        - statement
        - themes
        - status
        :raises: requests.HTTPError
        If any request to the OGCAPI endpoint fails.
        """
        all_collections = []
        # build initial request url
        initial_url = self.build_ogcapi_query_url()

        resp = self.session.get(initial_url, timeout=self.timeout)
        if resp.status_code != requests.codes.ok:
            logger.error(
                "Failed to fetch collections from ogcapi",
                status=resp.status_code,
                url=initial_url,
            )
            resp.raise_for_status()

        resp_dict = resp.json()
        total_collections = resp_dict.get("total", 0)
        logger.info(f"Total collections through OCGAPI: {total_collections}")

        all_collections.extend(self._parse_fetched_collection_data(resp_dict))

        search_after = resp_dict.get("search_after", None)
        if not search_after:
            logger.info("No search after")
            return pd.DataFrame(all_collections)

        current_page = 1
        logger.debug(f"Fetching current page {current_page} | url: {initial_url}")

        while search_after:
            # set a sleep time to avoid too frequent query
            time.sleep(self.sleep_time)
            encoded_search_after_filter = self.build_search_after_filter(search_after)
            query_url = initial_url + encoded_search_after_filter
            current_page += 1
            logger.debug(f"Fetching current page {current_page} | url: {query_url}")

            resp = self.session.get(query_url, timeout=self.timeout)

            if resp.status_code != requests.codes.ok:
                logger.error(
                    "Failed to fetch collections from ogcapi",
                    status=resp.status_code,
                    url=query_url,
                )
                resp.raise_for_status()

            resp_dict = resp.json()
            all_collections.extend(self._parse_fetched_collection_data(resp_dict))

            search_after = resp_dict.get("search_after", [])
        return pd.DataFrame(all_collections)
