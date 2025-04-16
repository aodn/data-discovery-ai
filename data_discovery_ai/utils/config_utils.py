from pathlib import Path
from data_discovery_ai.common.constants import PARAMETER_FILE
from typing import Dict, Any, TypedDict
import yaml


# the key should be restricted as defined
class ElasticsearchConfig(TypedDict):
    batch_size: int
    sleep_time: int
    es_index_name: str


class KeywordClassificationTrainerConfig(TypedDict):
    vocabs: list[str]
    test_size: float
    n_splits: int
    dropout: float
    learning_rate: float
    fl_gamma: float
    fl_alpha: float
    epoch: int
    batch_size: int
    early_stopping_patience: int
    reduce_lr_patience: int
    validation_split: float


class DeliveryClassificationTrainerConfig(TypedDict):
    test_size: float
    n_estimators: int
    threshold: float
    n_components: float


class TrainerConfig(TypedDict):
    keyword_classification: KeywordClassificationTrainerConfig
    delivery_classification: DeliveryClassificationTrainerConfig


class KeywordClassificationConfig(TypedDict):
    confidence: float
    top_N: int
    separator: str
    pretrained_model: str


class DescriptionFormattingConfig(TypedDict):
    model: str
    temperature: float
    max_tokens: int


class ModelConfig(TypedDict):
    keyword_classification: KeywordClassificationConfig
    description_formatting: DescriptionFormattingConfig
    trainer: TrainerConfig


class FullConfig(TypedDict):
    elasticsearch: ElasticsearchConfig
    model: ModelConfig


class ConfigUtil:
    def __init__(self, config_file: str = PARAMETER_FILE) -> None:
        self.base_dir = Path(__file__).resolve().parent.parent
        self.config_file = self.base_dir / "common" / config_file
        self._config_data: FullConfig = self._load_yaml()

    def _load_yaml(self) -> FullConfig:
        if not self.config_file.exists():
            raise FileNotFoundError(f"YAML config file not found: {self.config_file}")

        with open(self.config_file, "r") as f:
            return yaml.safe_load(f)

    def get_es_config(self) -> ElasticsearchConfig:
        return self._config_data["elasticsearch"]

    def get_keyword_classification_config(self) -> KeywordClassificationConfig:
        return self._config_data["model"]["keyword_classification"]

    def get_description_formatting_config(self) -> DescriptionFormattingConfig:
        return self._config_data["model"]["description_formatting"]

    def get_keyword_trainer_config(self) -> KeywordClassificationTrainerConfig:
        return self._config_data["model"]["trainer"]["keyword_classification"]

    def get_delivery_trainer_config(self) -> DeliveryClassificationTrainerConfig:
        return self._config_data["model"]["trainer"]["delivery_classification"]
