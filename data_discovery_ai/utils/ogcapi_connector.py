import math
import os
import time
from typing import Optional, List, Any
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3 import Retry

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
        build the encoded search after filter. The search_after value comes from the response.
        :param search_after:
        :return: encoded search after filter string
        """
        filter_value = self.FILTER_SEPARATOR.join(str(item) for item in search_after)
        filter_str = f"{self.SEARCH_AFTER_PARAM}{self.FILTER_PREFIX}{filter_value}{self.FILTER_SUFFIX}"
        encoded_filter = quote(filter_str, safe="")
        encoded_search_after_filter = self.QUERY_CONNECTOR + encoded_filter
        return encoded_search_after_filter

    def build_ogcapi_query_url(self):
        base_url = self.host.rstrip("/")
        endpoint = self.endpoint.lstrip("/")
        page_size_str = str(self.page_size)

        ogcapi_query_url = f"{base_url}/{endpoint}{self.query}{page_size_str}"

        logger.debug(f"OGCAPI query url: {ogcapi_query_url}")
        return ogcapi_query_url

    def _parse_fetched_collection_data(self, resp_dict: dict) -> List[dict]:
        parsed_collection_data = []
        for collection in resp_dict.get("collections", []):
            props = collection.get("properties", {})
            parsed_collection_data.append(
                {
                    "id": collection.get("id"),
                    "title": collection.get("title"),
                    "description": collection.get("description"),
                    "statement": props.get("statement"),
                    "themes": props.get("themes", []),
                }
            )
        return parsed_collection_data

    def get_all_collections(self):
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

        all_collections.extend(self._parse_fetched_collection_data(resp_dict))

        search_after = resp_dict.get("search_after", None)
        if not search_after:
            logger.info("No search after")
            return all_collections

        while search_after:
            # set a sleep time to avoid too frequent query
            time.sleep(self.sleep_time)
            # build loop reqeust
            num_of_page = math.ceil(total_collections / self.page_size)
            for i in range(1, num_of_page):
                encoded_search_after_filter = self.build_search_after_filter(
                    search_after
                )
                query_url = initial_url + encoded_search_after_filter
                logger.debug("fetch_collections_page", url=query_url)

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
                # set a sleep time to avoid too frequent query
                time.sleep(self.sleep_time)
                logger.debug(f"Finalise fetching collections on page {i}")
        return all_collections
