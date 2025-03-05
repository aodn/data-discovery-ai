# A decision-making rule-based model for grouping links into 'Data Access'/'Document'/'Python Notebook'/ 'Other' categories.
import json
import requests
from typing import Any, Dict, List
from itertools import product, permutations
from data_discovery_ai.common.constants import GROUPING_RULES

from data_discovery_ai import logger

def link_grouping_model(links: List[Dict[str, str]]) -> List[Dict[str, str]]:
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

def content_keyword():
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
    href = link["href"].lower()
    title = link["title"].lower()

    for group, conditions in GROUPING_RULES.items():
        if "href" in conditions and any(keyword in href.lower() for keyword in conditions["href"]):
            return group

        if "title" in conditions and any(keyword in title.lower() for keyword in conditions["title"]):
            return group

        if "rel" in conditions and any(keyword in title.lower() for keyword in conditions["rel"]):
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