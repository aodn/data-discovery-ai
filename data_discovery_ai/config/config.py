import os
import logging
from pathlib import Path
from typing import Any, Dict, TypedDict, List
from dataclasses import dataclass, field
import yaml

import data_discovery_ai.config.constants as constants


@dataclass(frozen=True)
class MlflowConfig:
    port: int
    gateway: str


@dataclass(frozen=True)
class ElasticsearchConfig:
    batch_size: int
    sleep_time: int
    es_index_name: str


@dataclass(frozen=True)
class KeywordClassificationTrainerConfig:
    vocabs: List[str]
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
    rare_label_threshold: int
    separator: str


@dataclass(frozen=True)
class DeliveryClassificationTrainerConfig:
    test_size: float
    n_estimators: int
    threshold: float
    n_components: float
    separator: str
    max_depth: int
    max_leaf_nodes: int
    k_best: int
    max_iter: int


@dataclass(frozen=True)
class TrainerConfig:
    keyword_classification: KeywordClassificationTrainerConfig
    delivery_classification: DeliveryClassificationTrainerConfig


@dataclass(frozen=True)
class KeywordClassificationConfig:
    confidence: float
    top_N: int
    separator: str
    pretrained_model: str
    response_key: str


@dataclass(frozen=True)
class DescriptionFormattingConfig:
    model: str
    temperature: float
    max_tokens: int
    response_key: str


@dataclass(frozen=True)
class DeliveryClassificationConfig:
    pretrained_model: str
    separator: str
    response_key: str


@dataclass(frozen=True)
class SupervisorConfig:
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    supervisor: SupervisorConfig
    keyword_classification: KeywordClassificationConfig
    description_formatting: DescriptionFormattingConfig
    delivery_classification: DeliveryClassificationConfig
    link_grouping: Dict[str, Any]


@dataclass(frozen=True)
class FullConfig:
    elasticsearch: ElasticsearchConfig
    model: ModelConfig
    trainer: TrainerConfig
    logging: Dict[str, Any]


class ConfigUtil:
    """
    Configuration utility that loads settings in the order:
      1. ENV vars (mapped via ENV_VARS)
      2. YAML file (PARAMETER_FILE)
      3. Hardcoded DEFAULTS
    Also sets up logging according to configuration.
    """

    DEFAULTS: Dict[str, Any] = {
        "elasticsearch": {
            "es_index_name": "default_index",
            "batch_size": 100,
            "sleep_time": 5,
        },
        "logging": {"level": "INFO"},
    }

    # Default parameters overridable via environment vars
    ENV_VARS: Dict[str, str] = {
        "elasticsearch.batch_size": "ES_BATCH_SIZE",
        "elasticsearch.sleep_time": "ES_SLEEP_TIME",
        "elasticsearch.es_index_name": "ES_INDEX_NAME",
        "model.description_formatting.model": "DESCRIPTION_MODEL",
        # environment selection
        "environment": "PROFILE",
        # logging level
        "logging.level": "LOG_LEVEL",
    }

    def __init__(self, config_file: str = constants.PARAMETER_FILE) -> None:
        """Load YAML config, determine environment, and initialize logging."""
        self.base_dir = Path(__file__).resolve().parent.parent
        self.config_path = self.base_dir / "config" / config_file
        self._config_data: Dict[str, Any] = self._load_yaml()

        # determine environment (default to 'development')
        env_val = os.getenv(self.ENV_VARS["environment"])
        self.env = env_val.lower() if env_val else "development"

        # setup logging level
        level_str = self._get_value("logging.level", self.DEFAULTS["logging"]["level"])
        numeric_level = getattr(logging, level_str.upper(), logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

    def _load_yaml(self) -> Dict[str, Any]:
        """Read and parse the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _get_value(self, path: str, default: Any) -> Any:
        env_key = self.ENV_VARS.get(path)
        if env_key:
            env_val = os.getenv(env_key)
            if env_val is not None:
                try:
                    return type(default)(env_val)
                except (TypeError, ValueError) as e:
                    logging.warning(
                        f"Could not cast env var {env_key}='{env_val}' to {type(default)}. Error message: \n{e}"
                    )
                    return default

        parts = path.split(".")
        data: Any = self._config_data
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                return default
        return data

    def get_es_config(self) -> ElasticsearchConfig:
        sub = "elasticsearch"
        return ElasticsearchConfig(
            batch_size=self._get_value(
                f"{sub}.batch_size", self.DEFAULTS[sub]["batch_size"]
            ),
            sleep_time=self._get_value(
                f"{sub}.sleep_time", self.DEFAULTS[sub]["sleep_time"]
            ),
            es_index_name=self._get_value(
                f"{sub}.es_index_name", self.DEFAULTS[sub]["es_index_name"]
            ),
        )

    def get_supervisor_config(self) -> SupervisorConfig:
        return SupervisorConfig(
            settings=self._config_data.get("model", {}).get("supervisor", {})
        )

    def get_keyword_classification_config(self) -> KeywordClassificationConfig:
        m = self._config_data.get("model", {}).get("keyword_classification", {})
        return KeywordClassificationConfig(
            confidence=m.get("confidence", 0.0),
            top_N=m.get("top_N", 0),
            separator=m.get("separator", ""),
            pretrained_model=m.get("pretrained_model", ""),
            response_key=m.get("response_key", ""),
        )

    def get_description_formatting_config(self) -> DescriptionFormattingConfig:
        if self.env == "development":
            defaults = {
                "model": "llama3",
                "temperature": 0.0,
                "max_tokens": 4000,
                "response_key": "summaries.ai:description",
            }
        else:
            defaults = {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 10000,
                "response_key": "summaries.ai:description",
            }
        model = os.getenv(
            self.ENV_VARS["model.description_formatting.model"], defaults["model"]
        )
        return DescriptionFormattingConfig(
            model=model,
            temperature=defaults["temperature"],
            max_tokens=defaults["max_tokens"],
            response_key=defaults["response_key"],
        )

    def get_delivery_classification_config(self) -> DeliveryClassificationConfig:
        m = self._config_data.get("model", {}).get("delivery_classification", {})
        return DeliveryClassificationConfig(
            pretrained_model=m.get("pretrained_model", ""),
            separator=m.get("separator", ""),
            response_key=m.get("response_key", ""),
        )

    def get_link_grouping_config(self) -> Dict[str, Any]:
        return self._config_data.get("model", {}).get("link_grouping", {})

    def get_keyword_trainer_config(self) -> KeywordClassificationTrainerConfig:
        tr = self._config_data.get("trainer", {}).get("keyword_classification", {})
        return KeywordClassificationTrainerConfig(**tr)

    def get_delivery_trainer_config(self) -> DeliveryClassificationTrainerConfig:
        tr = self._config_data.get("trainer", {}).get("delivery_classification", {})
        return DeliveryClassificationTrainerConfig(**tr)

    def get_mlflow_config(self) -> MlflowConfig:
        c = self._config_data.get("mlflow", {})
        return MlflowConfig(port=c.get("port", 53000), gateway=c.get("gateway", ""))
