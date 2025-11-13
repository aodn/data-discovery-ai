# the agent model for link grouping task
import json
from typing import Dict, Any, List, Tuple
from itertools import product, permutations
import requests
import re
import structlog

from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.enum.agent_enums import AgentType


logger = structlog.get_logger(__name__)


def subgroup_access_link(link: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refine the `ai:group` classification for links tagged as "Data Access".

    Inspects the link URL and adds a more specific subgroup in the format: "Data Access > [subgroup]". If no subgroup pattern is found, returns the link unchanged.
    Applied subgroups:
        - wfs (for WFS links)
        - wms (for WMS links)
        - thredds (for links pointing to THREDDS access)
        - aws (for links pointing to AWS Open Data Program registry)
    Input:
        link (Dict[str, Any]): A single AI-classified link dictionary.
    Output:
        Dict[str, Any]: The same link dictionary, with `ai:group` modified to include the subgroup if a match is found; otherwise unchanged.
    """
    sep = " > "

    if "wms" in link["rel"]:
        link["ai:group"] = link["ai:group"] + sep + "wms"
        return link
    if "wfs" in link["rel"]:
        link["ai:group"] = link["ai:group"] + sep + "wfs"
        return link

    subgroup_pattern = [
        # identifier for aws open data program registry: "registry.opendata.aws" in href
        ("aws", re.compile(r"\bregistry\.opendata\.aws\b", re.IGNORECASE)),
        # identifier for thredds: "thredds" in href
        ("thredds", re.compile(r"\bthredds\b", re.IGNORECASE)),
        # identifier "wfs" in href
        ("wfs", re.compile(r"(\bwfs\b|service\s*=\s*wfs\b)", re.IGNORECASE)),
        # identifier "wms" in href
        ("wms", re.compile(r"(\bwms\b|service\s*=\s*wms\b)", re.IGNORECASE)),
    ]
    href = link.get("href")
    matches: List[str] = [name for name, pat in subgroup_pattern if pat.search(href)]
    if not matches:
        return link

    parts = [p.strip() for p in (link.get("ai:group") or "Data Access").split(sep)]
    for sg in matches:
        if sg not in parts:
            parts.append(sg)

    link["ai:group"] = sep.join(parts)
    return link


class LinkGroupingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = AgentType.LINK_GROUPING.value
        self.config = ConfigUtil.get_config()
        self.model_config = self.config.get_link_grouping_config()
        self.supervisor = None

    def set_supervisor(self, supervisor):
        self.supervisor = supervisor

    def set_required_fields(self, required_fields):
        return super().set_required_fields(required_fields)

    def is_valid_request(self, request: Dict[str, str]) -> bool:
        return super().is_valid_request(request)

    def make_decision(self, request: Dict[str, Any]) -> List[Any]:
        """
        The agent makes decision based on the request. The link grouping task only executes if the request is valid and the links are need to be grouped (the value of `rel` is not excluded).
        Input:
            request (Dict[str, Any]): The request format, which is expected to contain the following fields:
                links (List[Dict[str, str]]): A list of dictionaries. Each link is a dict with keys 'href', 'rel', 'type', and 'title'. An example likes:
                {
                    "href": "https://nbviewer.org/github/aodn/aodn_cloud_optimised/blob/main/notebooks/vessel_trv_realtime_qc.ipynb",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Access to Jupyter notebook to query Cloud Optimised converted dataset"
                }

        Output:
            List[Dict[str, str]]: the related links with full title and href for further grouping.
            Or an empty list if there are no valid links.
        """
        exclude_rel_values = self.model_config.get("exclude_rules", {}).get("rel", [])
        if self.is_valid_request(request):
            valid_links = []
            # check if the links are related
            links = request.get("links", [])
            for link in links:
                if (
                    link.get("rel") not in exclude_rel_values
                    and link.get("href")
                    and link.get("title")
                ):
                    valid_links.append(link)
            return valid_links
        else:
            return []

    def is_excluded(self, link: Dict[str, Any]) -> bool:
        exclude_rules = self.model_config.get("exclude_rules", {})

        for field, exclude_values in exclude_rules.items():
            field_value = link.get(field)
            if field == "title" and field_value is not None:
                field_value = parse_combined_title(field_value)[0]
            if field_value in exclude_values:
                return True

        return False

    def take_action(self, links: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not links:
            return []
        page_content_keywords = self.content_keyword()
        for link in links:
            keys = set(link.keys())
            if "href" not in keys or "title" not in keys:
                continue

            if self.is_excluded(link):
                continue

            link_group = self.grouping(link, page_content_keywords)
            if not link_group:
                continue

            link["ai:group"] = link_group

            # Add subgroup for Data Access link. Inspects the link URL and adds a more specific subgroup in the format: "Data Access > [subgroup]".
            if link_group == "Data Access":
                link = subgroup_access_link(link)

            if link_group == "Python Notebook":
                # make sure the python notebook type is as required
                link["type"] = "application/x-ipynb+json"
        return links

    def content_keyword(self) -> List[str]:
        """
        Given a list of keywords set, generate all possible combinations and permutations of the keywords. For example, given ['data', 'dataset'] and ['access'], the function will return:
        ['data access', 'dataset access', 'access data', 'access dataset']. These keywords are used to filter the content of the link page.
        Output:
            List[str]. A list of strings. Each string is a combination of keywords.
        """
        keyword_groups = self.model_config["grouping_rules"]["Data Access"]["content"]
        combinations = [" ".join(combo) for combo in product(*keyword_groups)]
        final_combinations = set()
        for phrase in combinations:
            words = phrase.split()
            final_combinations.update(" ".join(p) for p in permutations(words))

        # Convert set to list
        final_combinations = list(final_combinations)
        return final_combinations

    def grouping(self, link: Dict[str, str], page_content_keywords: List) -> str:
        """
        Based on the decision-making rules defined in grouping rules, determine the group of the link.
        Input:
            link: Dict[str, str]. A dictionary with keys 'href' and 'title'. The link to be grouped.
            page_content_keywords: List[str]. A list of strings. Each string is a combination of keywords. These keywords are used to filter the content of the link page.
        Output:
            str. The group of the link. It can be 'Data Access'/'Document'/'Python Notebook'/ 'Other'.
        """
        href = link.get("href", "").lower()
        combined_title = link.get("title", "").lower()
        title = parse_combined_title(combined_title)[0]
        rel = link.get("rel", "").lower()

        for group, conditions in self.model_config["grouping_rules"].items():
            if "href" in conditions and any(
                keyword in href for keyword in conditions["href"]
            ):
                return group

            if "title" in conditions and any(
                keyword in title for keyword in conditions["title"]
            ):
                return group

            if "rel" in conditions and any(
                keyword in rel for keyword in conditions["rel"]
            ):
                return group

        # if no condition met, crawl the content to check if keywords are present
        try:
            # sometimes rel is not 100% accurate, we use a more restrictive rule to detect a data link for rel=data links with exact data-like urls
            # e.g. {
            #             "href": "https://mnf.csiro.au/",
            #             "rel": "data",
            #             "type": "",
            #             "title": "Marine National Facility"
            #         }
            data_path_keywords = ["data", "dataset", "download", "file", "api"]
            is_data_url = any(keyword in href.lower() for keyword in data_path_keywords)

            if rel == "data" and not href.endswith(".html") and is_data_url:
                return "Data Access"
            if href.endswith(".html"):
                resp = requests.get(href, timeout=(3, 5))
                if resp.status_code == 200:
                    content = resp.text.lower()
                    if any(keyword in content for keyword in page_content_keywords):
                        return "Data Access"
        except requests.exceptions.RequestException:
            logger.error(f"Failed to crawl the link: {link['href']}")
            return "Other"

        return "Other"

    def execute(self, request: Dict[str, Any]) -> None:
        flag = self.make_decision(request)
        if not flag:
            self.response = {self.model_config["response_key"]: []}
        else:
            links = request["links"]
            grouped_links = self.take_action(links)
            self.response = {self.model_config["response_key"]: grouped_links}

        logger.debug(f"{self.type} agent finished, it responses: \n {self.response}")


def parse_combined_title(combined_title: str) -> tuple[str | None, str | None]:
    """
    Helper function to parse combined text in link.title field, which is a combination of link title and description in the json format {"title": "My Title", "description": "My Description"}.
    The description field must exist in the JSON (even if empty) to parse the title. Otherwise, return the original text.

    :param combined_title: str: the combined json string in link.title field
    :return: tuple[str | None, str | None]. The parsed title and description in string. description can be None if it's an empty string.
    """
    # return None, None for both title and description if the combined text is None or empty
    if combined_title is None or combined_title.strip() == "":
        return None, None

    # Try to parse as JSON
    try:
        data = json.loads(combined_title)

        if not isinstance(data, dict):
            return combined_title.strip(), None

        # Check if description field exists (key must be present)
        if "description" not in data:
            # No description field, return original text
            return combined_title.strip(), None

        # Parse title
        title = data.get("title")
        if isinstance(title, str):
            title = title.strip()
            title = title if title else None
        else:
            title = None

        # Parse description
        description = data.get("description")
        if isinstance(description, str):
            description = description.strip()
            description = description if description else None
        else:
            description = None

        return title, description

    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, return original text as title with None description
        return combined_title.strip(), None
