import os
import logging
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass, field
import yaml

from data_discovery_ai.config.constants import PARAMETER_FILE


class EnvType(Enum):
    DEV = "development"
    EDGE = "edge"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass(frozen=True)
class MlflowConfig:
    port: int
    gateway: str


@dataclass(frozen=True)
class ApplicationConfig:
    port: int
    reload: bool


@dataclass(frozen=True)
class ElasticsearchConfig:
    batch_size: int
    sleep_time: int
    es_index_name: str
    es_ai_index_name: str


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
    threshold: float
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

    LOGLEVEL = "DEBUG"

    def __init__(self, config_file: str) -> None:
        """Load YAML config, determine environment, and initialize logging."""
        self.base_dir = Path(__file__).resolve().parent.parent
        self.config_file = self.base_dir / "config" / config_file
        self._config_data = self._load_yaml(self.config_file)

        self.config_file = self.base_dir / "config" / PARAMETER_FILE
        self._parameter_data = self._load_yaml(self.config_file)

        # determine environment (default to 'development')
        env_val = os.getenv("PROFILE")
        self.env = env_val.lower() if env_val else "development"

        # setup logging level
        self.init_logging()

    def init_logging(self):
        level_str = getattr(self, "LOGLEVEL", "DEBUG")
        numeric_level = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)

    @staticmethod
    def get_config(profile: EnvType = None):
        if profile is None:
            profile = EnvType(os.getenv("PROFILE", EnvType.DEV))

        match profile:
            case EnvType.PRODUCTION:
                return ProdConfig()

            case EnvType.EDGE:
                return EdgeConfig()

            case EnvType.STAGING:
                return StagingConfig()

            case _:
                return DevConfig()

    def _load_yaml(self, config_file: Path) -> Dict[str, Any]:
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, "r") as f:
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
        data: Any = self._parameter_data
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                return default
        return data

    def get_es_config(self) -> ElasticsearchConfig:
        sub = "elasticsearch"
        es_config = self._config_data.get(sub)
        return ElasticsearchConfig(
            batch_size=es_config.get("batch_size", 100),
            sleep_time=es_config.get("sleep_time", 1),
            es_index_name=es_config.get("es_index_name", ""),
            es_ai_index_name=es_config.get("es_ai_index_name", ""),
        )

    def get_supervisor_config(self) -> SupervisorConfig:
        return SupervisorConfig(
            settings=self._parameter_data.get("model", {}).get("supervisor", {})
        )

    def get_keyword_classification_config(self) -> KeywordClassificationConfig:
        m = self._parameter_data.get("model", {}).get("keyword_classification", {})
        return KeywordClassificationConfig(
            confidence=m.get("confidence", 0.0),
            top_N=m.get("top_N", 0),
            separator=m.get("separator", ""),
            pretrained_model=m.get("pretrained_model", ""),
            response_key=m.get("response_key", ""),
        )

    def get_description_formatting_config(self) -> DescriptionFormattingConfig:
        m = self._config_data.get("model", {}).get("description_formatting", {})
        return DescriptionFormattingConfig(
            model=m.get("model", "llama3"),
            temperature=m.get("temperature", 0.0),
            max_tokens=m.get("max_tokens", 4000),
            response_key=m.get("response_key", "summaries.ai:description"),
        )

    def get_delivery_classification_config(self) -> DeliveryClassificationConfig:
        m = self._parameter_data.get("model", {}).get("delivery_classification", {})
        return DeliveryClassificationConfig(
            pretrained_model=m.get("pretrained_model", ""),
            separator=m.get("separator", ""),
            response_key=m.get("response_key", ""),
        )

    def get_link_grouping_config(self) -> Dict[str, Any]:
        return self._parameter_data.get("model", {}).get("link_grouping", {})

    def get_keyword_trainer_config(self) -> KeywordClassificationTrainerConfig:
        tr = self._parameter_data.get("trainer", {}).get("keyword_classification", {})
        return KeywordClassificationTrainerConfig(**tr)

    def get_delivery_trainer_config(self) -> DeliveryClassificationTrainerConfig:
        tr = self._parameter_data.get("trainer", {}).get("delivery_classification", {})
        return DeliveryClassificationTrainerConfig(**tr)

    def get_mlflow_config(self) -> MlflowConfig:
        c = self._config_data.get("mlflow", {})
        return MlflowConfig(port=c.get("port", 53000), gateway=c.get("gateway", ""))

    def get_application_config(self) -> ApplicationConfig:
        c = self._config_data.get("application", {})
        port = c.get("port", 8000)
        reload_val = c.get("reload", False)
        is_reload = str(reload_val).lower() == "true"
        return ApplicationConfig(port=port, reload=is_reload)


class DevConfig(ConfigUtil):
    LOGLEVEL = "DEBUG"

    def __init__(self):
        config_file = "config-dev.yaml"
        super().__init__(config_file)


class EdgeConfig(ConfigUtil):
    LOGLEVEL = "INFO"

    def __init__(self):
        config_file = "config-edge.yaml"
        super().__init__(config_file)


class StagingConfig(ConfigUtil):
    LOGLEVEL = "INFO"

    def __init__(self):
        config_file = "config-staging.yaml"
        super().__init__(config_file)


class ProdConfig(ConfigUtil):
    LOGLEVEL = "WARNING"

    def __init__(self):
        config_file = "config-prod.yaml"
        super().__init__(config_file)
