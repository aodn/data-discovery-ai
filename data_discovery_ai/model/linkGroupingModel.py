# A decision-making rule-based model for grouping links into 'Data Access'/'Document'/'Python Notebook'/ 'Other' categories.
import json
import requests
from typing import Any, Dict, List
from itertools import product, permutations
from data_discovery_ai.common.constants import GROUPING_RULES

from data_discovery_ai import logger


def link_grouping_model(links: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Based on link href, title, and content shown on the link, group the links into 'Data Access'/'Document'/'Python Notebook'/ 'Other' categories.
    Input:
        links: List[Dict[str, str]]. A list of dictionaries. Each link is a dict with keys 'href', 'rel', 'type', and 'title'. An example likes:
        {
              "href": "https://nbviewer.org/github/aodn/aodn_cloud_optimised/blob/main/notebooks/vessel_trv_realtime_qc.ipynb",
              "rel": "related",
              "type": "text/html",
              "title": "Access to Jupyter notebook to query Cloud Optimised converted dataset"
            }
        Link grouping model only uses values in 'href' and 'title'. So if a link has no title or no href, it will be skipped.
    Output:
        List[Dict[str, str]]. A list of dictionaries. Each link is a dict with keys 'href', 'rel', 'type', 'title', and 'group'. An example likes:
        {
              "href": "https://nbviewer.org/github/aodn/aodn_cloud_optimised/blob/main/notebooks/vessel_trv_realtime_qc.ipynb",
              "rel": "related",
              "type": "text/html",
              "title": "Access to Jupyter notebook to query Cloud Optimised converted dataset",
              "group": "Data Access"
            }
        The group of each link is determined by the decision-making rules in constant GROUPING_RULES defined in data_discovery_ai/common/constants.py.
    """
    # catch empty links
    if not links or len(links) == 0:
        return []
    page_content_keywords = content_keyword()
    for link in links:
        # check the link is valid with href and title
        keys = set(link.keys())
        if "href" not in keys or "title" not in keys:
            logger.info(f"Invalid link with no href or title: {link}")
        else:
            link_group = make_decision(link, page_content_keywords)
            if link_group:
                link["group"] = link_group
    return links


def content_keyword() -> List[str]:
    """
    Given a list of keywords set, generate all possible combinations and permutations of the keywords. For example, given ['data', 'dataset'] and ['access'], the function will return:
    ['data access', 'dataset access', 'access data', 'access dataset']. These keywords are used to filter the content of the link page.
    Output:
        List[str]. A list of strings. Each string is a combination of keywords.
    """
    keyword_groups = GROUPING_RULES["Data Access"]["content"]
    combinations = [" ".join(combo) for combo in product(*keyword_groups)]
    final_combinations = set()
    for phrase in combinations:
        words = phrase.split()
        final_combinations.update(" ".join(p) for p in permutations(words))

    # Convert set to list
    final_combinations = list(final_combinations)
    return final_combinations


def make_decision(link: Dict[str, str], page_content_keywords: List) -> Dict[str, str]:
    """
    Based on the decision-making rules defined in GROUPING_RULES, determine the group of the link.
    Input:
        link: Dict[str, str]. A dictionary with keys 'href' and 'title'. The link to be grouped.
        page_content_keywords: List[str]. A list of strings. Each string is a combination of keywords. These keywords are used to filter the content of the link page.
    Output:
        str. The group of the link. It can be 'Data Access'/'Document'/'Python Notebook'/ 'Other'.
    """
    href = link["href"].lower()
    title = link["title"].lower()

    for group, conditions in GROUPING_RULES.items():
        if "href" in conditions and any(
            keyword in href.lower() for keyword in conditions["href"]
        ):
            return group

        if "title" in conditions and any(
            keyword in title.lower() for keyword in conditions["title"]
        ):
            return group

        if "rel" in conditions and any(
            keyword in title.lower() for keyword in conditions["rel"]
        ):
            return group

    # if no condition met, crawl the content to check if keywords are present
    try:
        resp = requests.get(link["href"], timeout=(3, 10))
        if resp.status_code == 200:
            logger.info(f"Crawling the link: {link['href']}")
            content = resp.text.lower()
            if any(keyword in content for keyword in page_content_keywords):
                return "Data Access"
    except:
        logger.info(f"Failed to crawl the link: {link['href']}")
        return "Other"

    return "Other"
