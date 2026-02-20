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


class HuggingfaceModel(Enum):
    NLI_MODEL_NAME = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    EMBEDDING_MODEL_NAME = "google-bert/bert-base-uncased"
