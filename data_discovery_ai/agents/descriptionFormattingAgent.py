# The agent model for description formatting task
import asyncio
import time
from ollama import chat

from typing import Dict
import re
import json

from data_discovery_ai import logger
from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil


def needs_formatting(abstract: str) -> bool:
    word_count = len(re.findall(r"\w+", abstract))
    return word_count > 200 and "\n" in abstract


def retrieve_json(model: str, output: str) -> str:
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
    if model == "gpt-4o-mini":
        return output.strip()

    # try directly parsing the JSON-like block first, it should be applied for all GPT-4o-mini model outputs
    match = re.search(r"\{.*?}", output, re.DOTALL)
    if not match:
        logger.error("No JSON found in LLM response.")
        return output.strip()

    json_str = match.group()
    try:
        parsed = json.loads(json_str)
        return parsed["formatted_abstract"]
    except json.JSONDecodeError as e:
        logger.error(f"No JSON found in LLM response. Error message: \n{e}")

    # if the first attempt fails, try to find a triple-quote JSON block, the llama often outputs in this way
    triple = re.search(
        r'\{\s*"formatted_abstract"\s*:\s*""".*?"""\s*}', output, re.DOTALL
    )
    if not triple:
        logger.error(
            f"No JSON found in LLM response. The original response is \n{output}"
        )
        return output.strip()

    block = triple.group()

    def esc(m):
        inner = m.group(1)
        return json.dumps(inner)

    fixed = re.sub(r'"""\s*(.*?)\s*"""', esc, block, flags=re.DOTALL)

    try:
        parsed = json.loads(fixed)
        return parsed["formatted_abstract"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"No JSON found in LLM response. Error message: \n{e}")
        return output.strip()


def chunk_text(text: str, max_length: int = 1200) -> list[str]:
    """
    Splits text into chunks of max_length characters, at paragraph or sentence boundaries.
    """
    import re

    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 < max_length:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


async def format_chunk_async(
    client, system_prompt, title, chunk, model, temp, max_tokens
):
    input_text = f"Title: \n{title} \nAbstract:\n{chunk}"
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        temperature=temp,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},  # Ensure JSON output
    )

    formatted = json.loads(completion.choices[0].message.content)
    return formatted["formatted_abstract"]


class DescriptionFormattingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "description_formatting"

        # load api key from .env file
        self.model_config = ConfigUtil.get_config().get_description_formatting_config()

        self.supervisor = None

    def set_supervisor(self, supervisor):
        self.supervisor = supervisor

    def set_required_fields(self, required_fields: list) -> None:
        super().set_required_fields(required_fields)

    def is_valid_request(self, request: Dict[str, str]) -> bool:
        return super().is_valid_request(request)

    def execute(self, request: Dict[str, str]) -> None:
        """
        Execute the action module of the Description Formatting Agent. The action is to reformat the abstract text into Markdown format.
        The agent perceives the request, and make decision based on the received request. If it decides to take action, it will call the LLM module to reformat the abstract text and set self response as the reformatted abstract.
        Otherwise, it will set self.response as the original abstract text.
        Input:
            request (Dict[str, str]): The request format.
        """
        flag = self.make_decision(request)
        if not flag and "abstract" in request:
            self.response = {self.model_config.response_key: request["abstract"]}
        elif not flag and "abstract" not in request:
            self.response = {self.model_config.response_key: ""}
        else:
            title = request["title"]
            abstract = request["abstract"]
            self.response = {
                self.model_config.response_key: self.take_action(title, abstract)
            }
        logger.info(f"{self.type} agent finished, it responses: \n {self.response}")

    def make_decision(self, request: Dict[str, str]) -> bool:
        """
        Make decision based on the abstract text. This includes two-step validation:
        1. check if the request is valid as contains all required field and,
        2. check if the abstract need to be reformatted. The agent model only takes action if the abstract is more than 200 words and has more than one paragraph.
        Input:
            request (Dict[str, str]): The request format, which is expected to contain the following fields:
                title (str): The title of the metadata.
                abstract (str): The abstract of the metadata to be reformatted.
        Output:
            bool: True if the agent takes action, False otherwise.
        """
        return self.is_valid_request(request) and needs_formatting(request["abstract"])

    async def take_action_async(self, title: str, abstract: str) -> str:
        """
        Processes long abstracts in parallel chunks for faster LLM formatting.
        """
        system_prompt = """
        Process a metadata record's title and abstract. Reformat the abstract using Markdown format with these requirements: 1. keeping original text. 2. Markdown format: Lists: Each item on new line, start with -. Headings: # H1, ## H2, ### H3, #### H4. Bold: **text**. Italics: *text*. Links: URLs (www/http/https) as [text](www/http/https).
        Your response should in the following JSON format:
        {
        "formatted_abstract": "[Markdown-formatted text]"
        }
        """
        try:
            client = self.supervisor.llm_client
            model = self.model_config.model
            temp = self.model_config.temperature
            max_tokens = self.model_config.max_tokens

            chunks = chunk_text(abstract, max_length=2000)

            tasks = [
                format_chunk_async(
                    client, system_prompt, title, chunk, model, temp, max_tokens
                )
                for chunk in chunks
            ]

            logger.info(f"Submitting {len(tasks)} concurrent chunk formatting tasks...")
            results = await asyncio.gather(*tasks)

            return "\n\n".join(results).strip()

        except Exception as e:
            logger.error(f"Error in calling LLM: {e}")
            return abstract

    def take_action(self, title: str, abstract: str) -> str:
        """
        Action module of the Description Formatting Agent. The task is to reformat the abstract text into Markdown format.
        Input:
            title (str): The title of the metadata.
            abstract (str): The abstract of the metadata to be reformatted.
        Output:
            Str: the markdown formatted abstract text.
            If the agent does not take action, the output is the original abstract text.
        """
        logger.info(f"Description is being reformatted by {self.type} agent")
        input_text = f"Title: \n{title} \nAbstract:\n{abstract}"
        response = None
        model = self.model_config.model
        try:
            # if you use OpenAI model in production or staging
            if model == "gpt-4o-mini":
                response = asyncio.run(self.take_action_async(title, abstract))
            elif model == "llama3":
                # in dev use free llama 3 model
                system_prompt = """
            Process a metadata record's title and abstract. Reformat the abstract, keeping original text, using Markdown: Lists: Each item on new line, start with -. Headings: # H1, ## H2, ### H3, #### H4. Bold: **text**. Italics: *text*. Links: URLs (www/http/https) as [text](www/http/https).
            Your response should in the following JSON format:
            {
            "formatted_abstract": "[Markdown-formatted text]"
            }
            """
                response = chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    options={
                        "temperature": self.model_config.temperature,
                        "max_tokens": self.model_config.max_tokens,
                    },
                )
                response = response.message.content
            return retrieve_json(model, response)
        except Exception as e:
            logger.error(f"Error in calling LLM: {e}")
            return abstract
