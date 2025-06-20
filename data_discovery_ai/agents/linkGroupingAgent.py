# the agent model for link grouping task
from typing import Dict, Any, List
from itertools import product, permutations
import requests

from data_discovery_ai import logger
from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil


class LinkGroupingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "link_grouping"
        self.config = ConfigUtil()
        self.model_config = self.config.get_link_grouping_config()

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

    def take_action(self, links: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not links:
            return []
        else:
            page_content_keywords = self.content_keyword()
            exclude_rules = self.model_config.get("exclude_rules", {})
            for link in links:
                keys = set(link.keys())
                if "href" not in keys or "title" not in keys:
                    logger.info(f"Invalid link with no href or title: {link}")
                    continue

                is_excluded = False
                for field, exclude_values in exclude_rules.items():
                    if link.get(field) in exclude_values:
                        is_excluded = True
                        break

                if is_excluded:
                    continue

                link_group = self.grouping(link, page_content_keywords)
                if link_group:
                    link["ai:group"] = link_group
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
        href = link["href"].lower()
        title = link["title"].lower()

        for group, conditions in self.model_config["grouping_rules"].items():
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
        except requests.exceptions.RequestException:
            logger.info(f"Failed to crawl the link: {link['href']}")
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

        logger.info(f"{self.type} agent finished, it responses: \n {self.response}")
