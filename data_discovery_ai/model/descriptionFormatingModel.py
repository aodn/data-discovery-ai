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
                self.openai_api_key = os.getenv("OPENAI_API_KEY")
                self.llm_model = "gpt-4o-mini"
            elif llm_tool == "llama":
                self.llm_model = "llama3"
            else:
                self.llm_model = None
        else:
            self.llm_model = None
        
        if not self.llm_model:
            raise ValueError(
                'Available model name: ["openai", "llama"]'
            )


    def description_reformatting(self, title, abstract):
        input_text = f"Title: \n{title} \nAbstract:\n{abstract}"
        system_prompt = """You are a Marine Science Officer processing metadata records. Given a title and abstract of a metadata record, perform the following tasks:

                            Task 1: Convert Text to Markdown
                            Reformat the text while preserving its orginal text content. Apply markdown identifiers if necessary:(1) if it is a list, each item should be on a new line, starting with a hyphen. (2) Heading 1 starts with \#, Heading 2 starts with \#\# Heading 3 starts with \#\#\#, Heading 4 starts with \#\#\#\#. (3) Bold text is enclosed in double asterisks. (4) Italics text is enclosed in single asterisks. (5) If the text is a link (starting with www or https or http), it should be enclosed in square brackets followed by parentheses with the URL in parentheses.

                            Task 2: Classify Data Delivery Mode
                            Identify the data delivery mode of the data described by the record. Choose from the options [Completed/Real-Time/Delayed/Other] based on the following criteria:
                            - Completed: The data is fully delivered.
                            - Real-Time: The data is not fully delivered, and updated in real time.
                            - Delayed: The data is not fully delivered, and updated delayed.
                            - Other: If it describes a tool, document, or model.

                            Task 3: Return JSON Output
                            Format the response as:

                            {
                            "formatted_abstract": "[Markdown-formatted text]",
                            "data_delivery_mode": "[Completed/Real-Time/Delayed/Other]"
                            }
                            """
        response = None
        if self.llm_model == "gpt-4o-mini":
            client = OpenAI(api_key=self.openai_api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                    ],
                temperature=0.1,
                max_tokens=10000)
            response = completion.choices[0].message.content
        elif self.llm_model == "llama3":
            response: ChatResponse = chat(model='llama3', messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_text}
                    ],
                    options={"temperature": 0.0,
                            "max_tokens": 4000}
                )
            response = response.message.content
        
        if response:
            return self.retrieve_json(response)


    def retrieve_json(self, output):
        # find json like text in the output
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                parsed_json = json.loads(json_str)
                return parsed_json
            except json.JSONDecodeError:
                logger.info("Error: JSONDecodeError, try to fix the JSON string")
                # replace three single quotes with double quotes
                json_str = json_str.replace('"""', "\"")
                # replace new line with \n
                json_str = json_str.replace('\n', '\\n')
                return json.loads(json_str)
        else:
            return None
    

    def is_valid_llm_tool(self, llm_tool: str) -> bool:
        """
        Check if the given LLM tool is valid.
        """
        return llm_tool in ["openai", "llama"]
