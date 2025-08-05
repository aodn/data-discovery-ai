from enum import Enum


class LlmModels(Enum):
    GPT = "gpt-4o-mini"
    OLLAMA = "llama3"


class AgentType(Enum):
    BASE = "base"
    SUPERVISOR = "supervisor"
    LINK_GROUPING = "link_grouping"
    DESCRIPTION_FORMATTING = "description_formatting"
    KEYWORD_CLASSIFICATION = "keyword_classification"
    DELIVERY_CLASSIFICATION = "delivery_classification"
