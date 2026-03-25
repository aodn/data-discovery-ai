import structlog
from typing import Dict, Any, List

from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.enum.agent_enums import AgentType
from data_discovery_ai.agents.linkGroupingAgent import parse_combined_title
import re
from urllib.parse import urlparse, parse_qs

logger = structlog.get_logger(__name__)


class DownloadableLinkAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = AgentType.DOWNLOADABLE_LINK_GROUPING
        self.config = ConfigUtil.get_config()
        self.model_config = self.config.get_downloadable_link_grouping_config()

        exclude = self.model_config.get("exclude_rules", {})
        self._exclude_title: set[str] = {v.lower() for v in exclude.get("title", [])}
        self._exclude_description: set[str] = {
            v.lower() for v in exclude.get("description", [])
        }
        self._exclude_group: set[str] = {v.lower() for v in exclude.get("ai:group", [])}

        grouping = self.model_config.get("grouping_rules", {})
        self._downloadable_groups: set[str] = {
            v.lower() for v in grouping.get("ai:group", [])
        }
        self._downloadable_href: set[str] = {
            v.lower() for v in grouping.get("href", [])
        }

    def execute(self, data_access_links: List[Dict[str, Any]]) -> None:
        ai_assets: Dict[str, Any] = {}

        for link in data_access_links:
            if self._is_downloadable(link):
                href = (link.get("href") or "").strip()
                if href:
                    # build asset dict
                    # the original link title is a combined json string, e.g., "title": "{\"title\":\"Project summary - Recreational Fisheries Databases\",\"description\":\"Project summary - Recreational Fisheries Databases\"}"
                    # we need to parse it to title and description field
                    combined_title_description = link.get("title")
                    title = parse_combined_title(combined_title_description)[0]
                    description = parse_combined_title(combined_title_description)[1]
                    ai_assets[href] = {
                        "href": link.get("href"),
                        "title": title,
                        "description": description if description else "",
                        "type": link.get("type", ""),
                        "role": "DOWNLOAD",
                    }

        self.response = {self.model_config["response_key"]: ai_assets}
        logger.debug(
            f"{self.type} agent finished, "
            f"found {len(ai_assets)} downloadable assets"
        )

    def _is_downloadable(self, link: Dict[str, Any]) -> bool:
        group = link.get("ai:group", "").lower()
        href = link.get("href", "").lower()
        title = link.get("title", "").lower()
        description = link.get("description", "").lower()
        downloadable_flag = link.get("ai:is_page_downloadable", None)

        if downloadable_flag:
            return True
        # exclude invalid href
        if not self._is_valid_href(href):
            return False

        # exclude wms group
        if group in self._exclude_group:
            return False
        # exclude visualisation link
        if any(kw in title for kw in self._exclude_title):
            return False

        # exclude survey link
        if any(kw in description for kw in self._exclude_description):
            return False

        # include wfs/thredds/aws groups
        if group in self._downloadable_groups:
            return True

        # fall back to identify with href
        if any(kw in href for kw in self._downloadable_href):
            return True

        return False

    def _is_valid_href(self, href: str) -> bool:
        """
        Check if href is a well-formed URL.
        Returns False for URLs with unencoded backslashes.
        """
        try:
            parsed = urlparse(href)

            if not parsed.scheme or not parsed.netloc:
                return False

            # Unencoded backslashes in any part of the URL
            if "\\" in href:
                return False

            if re.search(r"[A-Za-z]:[/\\]", parsed.query):
                return False

            return True

        except Exception:
            return False
