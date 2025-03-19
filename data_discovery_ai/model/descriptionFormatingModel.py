# The LLM-based agent model for reformatting description
from openai import OpenAI
import os
from dotenv import load_dotenv
from ollama import chat
from ollama import ChatResponse

import difflib
import re
import json

from data_discovery_ai import logger


class DescriptionFormatingAgent:
    def __init__(self, llm_tool):
        if self.is_valid_llm_tool(llm_tool):
            self.llm_tool = llm_tool
            if llm_tool == "openai":
                load_dotenv()
                self.openai_api_key = os.getenv("OPENAI_API_KEY")
                self.llm_model = "gpt-4o-mini"
            elif llm_tool == "llama":
                self.llm_model = "llama3"
            else:
                self.llm_model = None
        else:
            self.llm_model = None

        if not self.llm_model:
            raise ValueError('Available model name: ["openai", "llama"]')
        
        self.status = "active"
        
    def make_decision(self, abstract) -> bool:
        # only execute the action when the passing description is too long (over than 200 words and more than one paragraph)
        if len(abstract.split()) > 200 and len(abstract.split("\n")) > 1:
            return True
        else:
            self.status = "inactive"
            return False


    def take_action(self, title, abstract):
        if self.make_decision(abstract):
            logger.info("Descrition is being reformatted by DescriptionFormatingAgent")

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
        if self.llm_model == "gpt-4o-mini":
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
        elif self.llm_model == "llama3":
            response: ChatResponse = chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                ],
                options={"temperature": 0.0, "max_tokens": 4000},
            )
            response = response.message.content

        if response:
            return self.retrieve_json(response)
        else:
            return abstract
        

    def retrieve_json(self, output):
        # find json like text in the output
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                parsed_json = json.loads(json_str)
                return parsed_json["formatted_abstract"]
            except json.JSONDecodeError:
                logger.info("Error: JSONDecodeError, try to fix the JSON string")
                # replace three single quotes with double quotes
                json_str = json_str.replace('"""', '"')
                # replace new line with \n
                json_str = json_str.replace("\n", "\\n")
                return json.loads(json_str)["formatted_abstract"]
        else:
            return None

    def is_valid_llm_tool(self, llm_tool: str) -> bool:
        """
        Check if the given LLM tool is valid.
        """
        return llm_tool in ["openai", "llama"]
