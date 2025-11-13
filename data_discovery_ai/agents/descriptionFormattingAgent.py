# The agent model for description formatting task
import asyncio
from ollama import chat

from typing import Dict, Optional
import re
import json
import structlog

from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.enum.agent_enums import LlmModels, AgentType

logger = structlog.get_logger(__name__)


def needs_formatting(abstract: str) -> bool:
    word_count = len(re.findall(r"\w+", abstract))
    return word_count > 200 and "\n" in abstract


def _strip_trailing_punct(s: str) -> tuple[str, str]:
    """
    split punctuation in the end of email or link, so that to avoid them in the converted email/link.
    """
    trail_punct = ".,!?:;"
    trailing = ""
    while s and s[-1] in trail_punct:
        trailing = s[-1] + trailing
        s = s[:-1]
    if s.endswith(")") and s.count("(") < s.count(")"):
        trailing = ")" + trailing
        s = s[:-1]
    return s, trailing


def _wrap_url(m: re.Match) -> str:
    url = m.group("url")
    core, trailing = _strip_trailing_punct(url)
    display = core
    href = (
        core if core.lower().startswith(("http://", "https://")) else f"https://{core}"
    )
    return f"[{display}]({href}){trailing}"


def _wrap_email(m: re.Match) -> str:
    email = m.group("email")
    return f"[{email}](mailto:{email})"


def manual_wrapper_description(abstract: str) -> str:
    """
    This is the customized function for processing short descriptions (i.e., descriptions that do not need to be formatted—returns false from the `needs_formatting` function) to add Markdown tags for links and emails in the description.
    Input: abstract: str. The original description text.
    Output: str. The enhanced description text.
    """
    # mapping rule: detect urls with these rules: (1) Optional protocol prefix: http://, https://, or www; (2) Domain name format: must start and end with alphanumeric characters (a-z, A-Z, 0-9) within 61 characters max (to falls in DNS standard)
    url_pattern = r"(?P<url>(?:https?://|www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}(?:[^\s<>()]*)?)"
    email_pattern = r"(?P<email>[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})"

    combined_pattern = f"({email_pattern})|({url_pattern})"
    combined_re = re.compile(combined_pattern, flags=re.IGNORECASE)

    def combined_replacer(match):
        if match.group(1):
            email_match = re.match(email_pattern, match.group(1))
            return _wrap_email(email_match)
        elif match.group(2):
            url_match = re.match(url_pattern, match.group(2))
            return _wrap_url(url_match)
        return match.group(0)

    abstract = combined_re.sub(combined_replacer, abstract)

    return abstract


def retrieve_json(model: str, output: str) -> str:
    """
    Retrieve json from the output text. The output is expected to contained text in the format of:
    {
        "formatted_abstract": "[Markdown-formatted text]"
    }
    Input:
        output: str. The output text from the LLM model (GPT or OLLAMA).
    Output:
        parsed_json_str: str. The parsed json string from the output text, which is the value of "formatted_abstract" key. If parsed json string is not found or failed, return None.
    """
    if model == LlmModels.GPT.value:
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


def chunk_text(text: str, max_length: int = 1000) -> list[str]:
    """
    Splits text into chunks of max_length characters, at paragraph or sentence boundaries.
    """
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
    client,
    system_prompt,
    chunk,
    model,
    temp,
    max_tokens,
    chunk_index=None,
    previous_formatted_tail=None,
):
    """
    Format a single chunk with context information

    Args:
        client: LLM client
        system_prompt: Base system prompt
        chunk: Text chunk to format
        model: Model name
        temp: Temperature
        max_tokens: Max tokens
        chunk_index: Current chunk number (0-indexed), None for first chunk
        previous_formatted_tail: the last sentence of the previous chunk, None for the first chunk
    """
    try:
        if chunk_index is None or chunk_index == 0:
            input_text = build_user_prompt(chunk_text=chunk, previous_tail=None)
        else:
            input_text = build_user_prompt(
                chunk_text=chunk, previous_tail=previous_formatted_tail
            )

        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=temp,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        formatted = json.loads(completion.choices[0].message.content)

        if "formatted_abstract" not in formatted:
            raise KeyError("Response does not contain 'formatted_abstract' key")

        return formatted["formatted_abstract"]

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        raise RuntimeError(f"Error formatting chunk: {e}")


def build_system_prompt() -> str:
    """
    To create the system prompt used for llm
    Output: str. the system prompt
    """
    return (
        "Format text between <<<CHUNK_START>>> and <<<CHUNK_END>>> as plain markdown.\n"
        "Rules:\n"
        "- Preserve original wording.\n"
        "- NO additional headers unless the source explicitly has section labels (≤10 words).\n"
        "- NO emphasis: Do not use **bold**, *italic*, or any text emphasis unless the source explicitly has emphasis labels.\n"
        "- Bullet lists: use '-' per item if item are pure text. preserve nesting with 2 spaces.\n"
        "- Number lists: use numbered list '1' if item start with numbers like 1., (1), or 1)"
        "- [text](url), [email](mailto:email) for actual URLs/emails in source.\n"
        "- Keep one blank line between paragraphs.\n"
        "- Continue lists/sections across chunks if ongoing.\n"
        'Return your response as a JSON object with this structure: {"formatted_abstract": "<markdown>"}'
    )


def build_user_prompt(chunk_text: str, previous_tail: Optional[str]) -> str:
    """
    To create the user prompt used for llm to process abstract by chunks
    Input:
        chunk_index: int. Current chunk number (0-indexed), the index of the chunk in an abstract.
        chunk_text: str. Chunk text of the record.
        previous_tail: str. Chunk text of the previous chunk, None for the first chunk
    Output: str. the system prompt
    """
    if previous_tail is not None:
        previous_tail_block = f"""Previous formatted tail (context only, do not repeat):
<<<PREVIOUS_TAIL>>>
{previous_tail}
<<<END_PREVIOUS_TAIL>>>
"""
    else:
        previous_tail_block = ""

    chunk_block = f"""Below is the text to format. Process only content inside markers.
<<<CHUNK_START>>>
{chunk_text}
<<<CHUNK_END>>>"""

    return previous_tail_block + chunk_block


def extract_last_sentence(text: str) -> str:
    """
    Extract the last sentence from formatted text for context.
    Input: text: Formatted text

    Output: Last sentence of the text or last 100 characters if no sentence boundary found
    """
    if not text or not text.strip():
        return ""

    text = text.strip()
    sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]

    last_boundary = -1
    for ending in sentence_endings:
        pos = text.rfind(ending)
        if pos > last_boundary:
            last_boundary = pos

    if last_boundary != -1:
        return text[last_boundary + 2 :].strip()
    else:
        return text[-100:].strip() if len(text) > 100 else text


class DescriptionFormattingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = AgentType.DESCRIPTION_FORMATTING.value

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
            result = manual_wrapper_description(request["abstract"])
            self.response = {self.model_config.response_key: result}
        elif not flag and "abstract" not in request:
            self.response = {self.model_config.response_key: ""}
        else:
            title = request["title"]
            abstract = request["abstract"]

            result = manual_wrapper_description(abstract)
            self.response = {
                self.model_config.response_key: self.take_action(title, result)
            }
        logger.debug(f"{self.type} agent finished, it responses: \n {self.response}")

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

    async def take_action_async(self, abstract: str) -> str:
        """
        Processes long abstracts in parallel chunks for faster LLM formatting.
        """
        system_prompt = build_system_prompt()

        try:
            client = self.supervisor.llm_client
            model = self.model_config.model
            temp = self.model_config.temperature
            max_tokens = self.model_config.max_tokens

            chunks = chunk_text(abstract, max_length=1000)

            if len(chunks) == 1:
                result = await format_chunk_async(
                    client,
                    system_prompt,
                    chunks[0],
                    model,
                    temp,
                    max_tokens,
                    chunk_index=0,
                    previous_formatted_tail=None,
                )
                return result

            results = []
            previous_tail = None

            for i, chunk in enumerate(chunks):
                try:
                    if i == 0:
                        # First chunk - no previous tail
                        formatted = await format_chunk_async(
                            client,
                            system_prompt,
                            chunk,
                            model,
                            temp,
                            max_tokens,
                            chunk_index=0,
                            previous_formatted_tail=None,
                        )
                    else:
                        # Subsequent chunks - include previous tail
                        formatted = await format_chunk_async(
                            client,
                            system_prompt,
                            chunk,
                            model,
                            temp,
                            max_tokens,
                            chunk_index=i,
                            previous_formatted_tail=previous_tail,
                        )

                    results.append(formatted)
                    # Extract the last sentence for next chunk's context
                    previous_tail = extract_last_sentence(formatted)

                except Exception as e:
                    logger.error(f"Error processing chunk {i + 1}: {e}")
                    results.append(chunk)
                    previous_tail = extract_last_sentence(chunk)

            return "\n\n".join(results).strip()

        except Exception as e:
            logger.error(f"Error in take_action_async: {e}")
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
        logger.debug(f"Description is being reformatted by {self.type} agent")
        response = None
        model = self.model_config.model
        try:
            # if you use OpenAI model in production or staging
            if model == LlmModels.GPT.value:
                response = asyncio.run(self.take_action_async(abstract))
            elif model == LlmModels.OLLAMA.value:
                # in dev use free llama 3 model
                system_prompt = build_system_prompt()
                user_prompt = build_user_prompt(abstract, title)
                response = chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
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
