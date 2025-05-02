import os
import logging
from pathlib import Path
from typing import Any, Dict, TypedDict, List
import yaml
import data_discovery_ai.config.constants as constants


class ElasticsearchConfig(TypedDict):
    batch_size: int
    sleep_time: int
    es_index_name: str


class KeywordClassificationTrainerConfig(TypedDict):
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
    response_key: str


class DescriptionFormattingConfig(TypedDict):
    model: str
    temperature: float
    max_tokens: int
    response_key: str


class ModelConfig(TypedDict):
    supervisor: Dict[str, Any]
    keyword_classification: KeywordClassificationConfig
    description_formatting: DescriptionFormattingConfig
    link_grouping: Dict[str, Any]
    delivery_classification: DeliveryClassificationTrainerConfig
    trainer: TrainerConfig


class FullConfig(TypedDict):
    elasticsearch: ElasticsearchConfig
    model: ModelConfig
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

    # Defaule parameters overridable via environment vars
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
        self._config_data: FullConfig = self._load_yaml()

        # determine environment (default to 'development')
        env_val = os.getenv(self.ENV_VARS["environment"])
        self.env = env_val.lower() if env_val else "development"

        # setup logging level
        level_str = self._get_value("logging.level", self.DEFAULTS["logging"]["level"])
        numeric_level = getattr(logging, level_str.upper(), logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

    def _load_yaml(self) -> FullConfig:
        """Read and parse the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return data  # type: ignore

    def _get_value(self, path: str, default: Any) -> Any:
        env_key = self.ENV_VARS.get(path)
        if env_key:
            env_val = os.getenv(env_key)
            if env_val is not None:
                try:
                    return type(default)(env_val)
                except Exception:
                    logging.warning(
                        f"Could not cast env var {env_key}='{env_val}' to {type(default)}"
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
        """Return Elasticsearch settings."""
        return ElasticsearchConfig(
            batch_size=self._get_value(
                "elasticsearch.batch_size", self.DEFAULTS["elasticsearch"]["batch_size"]
            ),
            sleep_time=self._get_value(
                "elasticsearch.sleep_time", self.DEFAULTS["elasticsearch"]["sleep_time"]
            ),
            es_index_name=self._get_value(
                "elasticsearch.es_index_name",
                self.DEFAULTS["elasticsearch"]["es_index_name"],
            ),
        )

    def get_supervisor_config(self) -> Dict[str, Any]:
        """Return supervisor agent configuration from YAML."""
        return self._config_data["model"].get("supervisor", {})

    def get_keyword_classification_config(self) -> KeywordClassificationConfig:
        """Return keyword classification parameters from YAML."""
        return self._config_data["model"]["keyword_classification"]

    def get_description_formatting_config(self) -> DescriptionFormattingConfig:
        """
        Return description formatting config. Defaults are hard-coded per-environment;
        ignores YAML for param values except model name override via ENV.
        """
        # set defaults by environment
        if self.env == "development":
            defaults = {
                "model": "llama3",
                "temperature": 0.0,
                "max_tokens": 4000,
                "response_key": "formatted_abstract",
            }
        else:
            defaults = {
                "model": "openai",
                "temperature": 0.1,
                "max_tokens": 10000,
                "response_key": "formatted_abstract",
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

    def get_link_grouping_config(self) -> Dict[str, Any]:
        # set up through parameters.yaml
        return self._config_data["model"].get("link_grouping", {})

    def get_delivery_classification_config(self) -> DeliveryClassificationTrainerConfig:
        # set up through parameters.yaml
        return self._config_data["model"]["delivery_classification"]

    def get_keyword_trainer_config(self) -> KeywordClassificationTrainerConfig:
        # set up through parameters.yaml
        return self._config_data["model"]["trainer"]["keyword_classification"]

    def get_delivery_trainer_config(self) -> DeliveryClassificationTrainerConfig:
        # set up through parameters.yaml
        return self._config_data["model"]["trainer"]["delivery_classification"]
