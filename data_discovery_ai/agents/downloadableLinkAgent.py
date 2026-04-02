import structlog
from typing import Dict, Any, List
import re
from urllib.parse import urlparse

from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.enum.agent_enums import AgentType, LinkAIRole

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

    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        The execution method for downloadable link agent to tag a link.
        Input:
            - request: should contains field `link` which is the link need to be tagged,
                and `instruction` which is the instruction from link grouping agent (to tell a link is downloadable
                or not through page content.
        Output:
            - Dict[str, Any]: the link with an extra field `ai:role`, e.g.:
            {
                "href": "https://geoserver.imas.utas.edu.au/geoserver/seamap/wfs?version=1.0.0&request=GetFeature&typeName=SeamapAus_VIC_statewide_habitats_2023&outputFormat=SHAPE-ZIP",
                "rel": "wfs",
                "type": "",
                "title": "{"title":"SHAPE-ZIP","description":"DATA ACCESS - This OGC WFS service returns the data (Statewide marine habitats 2023) in Shapefile format"}",
                "ai:group": "Data Access > wfs",
                "ai:role": ["download"]
            }
        """
        link = request.get("link", {})

        if link.get("ai:is_page_downloadable", False):
            link[self.model_config["response_key"]] = [LinkAIRole.DOWNLOAD.value]
            # remove temp flag used only by sub agent
            link.pop("ai:is_page_downloadable", None)
        if self._is_downloadable(link):
            link[self.model_config["response_key"]] = [LinkAIRole.DOWNLOAD.value]
        return link

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
