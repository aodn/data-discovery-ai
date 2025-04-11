# The agent-based model for description formatting task
from openai import OpenAI
import os
from dotenv import load_dotenv

from typing import Dict
import re
import json

from data_discovery_ai import logger
from data_discovery_ai.model.baseAgent import BaseAgent


class DescriptionFormattingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "DescriptionFormatting"
        # set status to 1 as active
        self.set_status(1)

        # load api key from .env file
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.response = {}

    def execute(self, request: Dict[str, str]) -> None:
        """
        Execute the action module of the Description Formatting Agent. The action is to reformat the abstract text into markdown format.
        The agent perceives the request, and make decision based on the received request. If it decides to take action, it will call the LLM module to reformat the abstract text and set self response as the reformatted abstract.
        Otherwise, it will set self.response as the original abstract text.
        Input:
            request (Dict[str, str]): The request format.
        """
        flag = self.make_decision(request)
        if not flag:
            self.response = {"formatted_abstract": request["abstract"]}
        else:
            title = request["title"]
            abstract = request["abstract"]
            self.response = {"formatted_abstract": self.take_action(title, abstract)}
        # set status to 2 as finished
        logger.info(
            "DescriptionFormattingAgent finished, it responses: \n %s", self.response
        )
        self.set_status(2)

    def is_valid_request(self, request: Dict[str, str]) -> bool:
        return all(k in request for k in ["title", "abstract"])

    def needs_formatting(self, abstract: str) -> bool:
        word_count = len(re.findall(r"\w+", abstract))
        return word_count > 200 and "\n" in abstract

    def make_decision(self, request: Dict[str, str]) -> bool:
        """
        Make decision based on the abstract text. This includes two step validation:
        1. check if the request is valid as contains all required field and,
        2. check if the abstract need to be reformatted. The agent model only takes action if the abstract is more than 200 words and has more than one paragraph.
        Input:
            request (Dict[str, str]): The request format, which is expetected to contain the following fields:
                title (str): The title of the metadata.
                abstract (str): The abstract of the metadata to be reformatted.
        Output:
            Boolean: True if the agent takes action, False otherwise.
        """
        return self.is_valid_request(request) and self.needs_formatting(
            request["abstract"]
        )

    def take_action(self, title: str, abstract: str) -> str:
        """
        Action module of the Description Formatting Agent. The action is to reformat the abstract text into markdown format.
        Input:
            title (str): The title of the metadata.
            abstract (str): The abstract of the metadata to be reformatted.
        Output:
            Str: the markdown formatted abstract text.
            If the agent does not take action, the output is the original abstract text.
        """
        logger.info("Description is being reformatted by DescriptionFormattingAgent")

        input_text = f"Title: \n{title} \nAbstract:\n{abstract}"
        system_prompt = """You are a Marine Science Officer processing metadata records. Given a title and abstract of a metadata record, perform the following tasks:

                            Task 1: Convert Text to Markdown
                            Reformat the text while preserving its orginal text content. Apply markdown identifiers if necessary:(1) if it is a list, each item should be on a new line, starting with a hyphen. (2) Heading 1 starts with \#, Heading 2 starts with \#\# Heading 3 starts with \#\#\#, Heading 4 starts with \#\#\#\#. (3) Bold text is enclosed in double asterisks. (4) Italics text is enclosed in single asterisks. (5) If the text is a link (starting with www or https or http), it should be enclosed in square brackets followed by parentheses with the URL in parentheses.

                            Task 2: Return JSON Output
                            Format the response as:

                            {
                            "formatted_abstract": "[Markdown-formatted text]"
                            }
                            """
        response = None
        try:
            client = OpenAI(api_key=self.openai_api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                ],
                temperature=0.1,
                max_tokens=10000,
            )
            response = completion.choices[0].message.content
            return self.retrieve_json(response)
        except Exception as e:
            logger.error(f"Error in calling OpenAI API: {e}")
            return abstract

    def retrieve_json(self, output: str) -> str:
        """
        Retrieve json from the output text. The output is expected to contained text in the format of:
        {
            "formatted_abstract": "[Markdown-formatted text]"
        }
        Input:
            output: str. The output text from the LLM model (GPT-4o-mini).
        Output:
            parsed_json_str: str. The parsed json string from the output text, which is the value of "formatted_abstract" key. If parsed json string is not found or failed, return None.
        """
        logger.debug(f"LLM raw output: {output}")
        # find json like text in the output
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            logger.error("No JSON found in LLM response.")
            return output.strip()

        json_str = match.group()
        try:
            parsed = json.loads(json_str)
            return parsed["formatted_abstract"]
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            return output.strip()
