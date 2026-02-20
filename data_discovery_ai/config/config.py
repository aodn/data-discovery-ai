import os
import logging
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass, field
import yaml
import structlog
from dotenv import load_dotenv

from data_discovery_ai.config.constants import PARAMETER_FILE
from data_discovery_ai.enum.agent_enums import LlmModels, AgentType


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
class OgcapiConfig:
    host: str
    endpoint: str
    query: str
    page_size: int
    sleep_time: int
    max_retries: int
    timeout: int


@dataclass(frozen=True)
class ElasticsearchConfig:
    es_index_name: str
    es_ai_index_name: str


@dataclass(frozen=True)
class ApplicationConfig:
    port: int
    reload: bool
    max_timeout: int
    sse_interval: int


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
    entailment_high: float
    entailment_low: float
    conflict_high: float
    model_name: str
    max_length: int
    max_sentence: int


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
    ogcapi: OgcapiConfig
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
        load_dotenv()
        """Load YAML config, determine environment, and initialize logging."""
        self.base_dir = Path(__file__).resolve().parent.parent
        self.config_file = self.base_dir / "config" / config_file
        self._config_data = self._load_yaml(self.config_file)

        self.parameter_file = self.base_dir / "config" / PARAMETER_FILE
        self._parameter_data = self._load_yaml(self.parameter_file)

        # determine environment (default to 'development')
        env_val = os.getenv("PROFILE")
        self.env = env_val.lower() if env_val else "development"

    def set_logging_level(self):
        """
        In dev environment, log level is set to DEBUG, output in String format.
        In edge/staging/production environments, log level is set to INFO (for edge and staging) or WARNING (for production),
        output in JSON format.
        """
        level_str = getattr(self, "LOGLEVEL", "DEBUG")
        numeric_level = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)

        # set third party libraries' logging level
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("uvicorn").setLevel(logging.WARNING)

        if self.env == "development":
            # set up logging for local development environment
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.dev.ConsoleRenderer(colors=False),
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
        else:
            self._init_json_logging()

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

    def get_log_config_path(self):
        """
        Get uvicorn log config path based on environment.

        Returns:
            str: Path to log_config.yaml for DEV
            None: For PROD/STAGING/EDGE (use structlog JSON)
        """
        if self.env == "development":
            log_config_path = self.base_dir / "log_config.yaml"
            return str(log_config_path) if log_config_path.exists() else None
        else:
            return None

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

    def _init_json_logging(self):
        """
        logging config to output in json format, example format:
            {
              "instant":"2025-06-06T00:01:44.529Z",
              "level":"INFO",
              "loggerName":"au.org.aodn.esindexer.BaseTestClass",
              "message":"Triggered indexer successfully",
              "endOfBatch":false,
              "threadId":1,
              "threadPriority":5,
              "service":"es-indexer"
            }
        """
        self.log_config_path = None

        logging.root.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.root.addHandler(handler)

        # add key-value mapping
        def add_service_name(logger, method_name, event_dict):
            event_dict["service"] = "data-discovery-ai"
            return event_dict

        def rename_timestamp(logger, method_name, event_dict):
            if "timestamp" in event_dict:
                event_dict["instant"] = event_dict.pop("timestamp")
            return event_dict

        def rename_logger_name(logger, method_name, event_dict):
            if "logger" in event_dict:
                event_dict["loggerName"] = event_dict.pop("logger")
            return event_dict

        def add_thread_info(logger, method_name, event_dict):
            """Add thread ID and priority information"""
            import threading

            thread = threading.current_thread()
            event_dict["threadId"] = thread.ident
            event_dict["threadPriority"] = 5  # use default priority
            return event_dict

        def add_logger_name(logger, method_name, event_dict):
            event_dict["loggerName"] = (
                logger.name if hasattr(logger, "name") else __name__
            )
            return event_dict

        def add_end_of_batch(logger, method_name, event_dict):
            event_dict["endOfBatch"] = False
            return event_dict

        structlog.configure(
            processors=[
                # instant field (timestamp use UTC timezone)
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                rename_timestamp,
                # level field
                structlog.stdlib.add_log_level,
                # loggerName field
                structlog.stdlib.add_logger_name,
                rename_logger_name,
                # message field
                structlog.processors.EventRenamer("message"),
                # endOfBatch field
                add_end_of_batch,
                # threadId and threadPriority fields
                add_thread_info,
                # service field
                add_service_name,
                # in JSON format
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def get_es_config(self) -> ElasticsearchConfig:
        sub = "elasticsearch"
        es_config = self._config_data.get(sub)
        return ElasticsearchConfig(
            es_index_name=es_config.get("es_index_name", ""),
            es_ai_index_name=es_config.get("es_ai_index_name", ""),
        )

    def get_ogcapi_config(self) -> OgcapiConfig:
        sub = "ogcapi"
        ogcapi_config = self._config_data.get(sub)
        return OgcapiConfig(
            host=ogcapi_config.get("host", ""),
            endpoint=ogcapi_config.get("endpoint", ""),
            query=ogcapi_config.get("query", ""),
            page_size=ogcapi_config.get("page_size", 1),
            sleep_time=ogcapi_config.get("sleep_time", 1),
            max_retries=ogcapi_config.get("max_retries", 1),
            timeout=ogcapi_config.get("timeout", 1),
        )

    def get_supervisor_config(self) -> SupervisorConfig:
        return SupervisorConfig(
            settings=self._parameter_data.get("model", {}).get("supervisor", {})
        )

    def get_keyword_classification_config(self) -> KeywordClassificationConfig:
        m = self._parameter_data.get("model", {}).get(
            AgentType.KEYWORD_CLASSIFICATION.value, {}
        )
        return KeywordClassificationConfig(
            confidence=m.get("confidence", 0.0),
            top_N=m.get("top_N", 0),
            separator=m.get("separator", ""),
            pretrained_model=m.get("pretrained_model", ""),
            response_key=m.get("response_key", ""),
        )

    def get_description_formatting_config(self) -> DescriptionFormattingConfig:
        m = self._config_data.get("model", {}).get(
            AgentType.DESCRIPTION_FORMATTING.value, {}
        )
        return DescriptionFormattingConfig(
            model=m.get("model", LlmModels.OLLAMA.value),
            temperature=m.get("temperature", 0.0),
            max_tokens=m.get("max_tokens", 4000),
            response_key=m.get("response_key", "summaries.ai:description"),
        )

    def get_delivery_classification_config(self) -> DeliveryClassificationConfig:
        m = self._parameter_data.get("model", {}).get(
            AgentType.DELIVERY_CLASSIFICATION.value, {}
        )
        return DeliveryClassificationConfig(
            pretrained_model=m.get("pretrained_model", ""),
            separator=m.get("separator", ""),
            response_key=m.get("response_key", ""),
        )

    def get_link_grouping_config(self) -> Dict[str, Any]:
        return self._parameter_data.get("model", {}).get(
            AgentType.LINK_GROUPING.value, {}
        )

    def get_keyword_trainer_config(self) -> KeywordClassificationTrainerConfig:
        tr = self._parameter_data.get("trainer", {}).get(
            AgentType.KEYWORD_CLASSIFICATION.value, {}
        )
        return KeywordClassificationTrainerConfig(**tr)

    def get_delivery_trainer_config(self) -> DeliveryClassificationTrainerConfig:
        tr = self._parameter_data.get("trainer", {}).get(
            AgentType.DELIVERY_CLASSIFICATION.value, {}
        )
        return DeliveryClassificationTrainerConfig(**tr)

    def get_mlflow_config(self) -> MlflowConfig:
        c = self._config_data.get("mlflow", {})
        return MlflowConfig(port=c.get("port", 53000), gateway=c.get("gateway", ""))

    def get_application_config(self) -> ApplicationConfig:
        c = self._config_data.get("application", {})
        port = c.get("port", 8000)
        reload_val = c.get("reload", False)
        is_reload = str(reload_val).lower() == "true"
        max_timeout = c.get("max_timeout", 60)
        sse_interval = c.get("sse_interval", 10)
        return ApplicationConfig(
            port=port,
            reload=is_reload,
            max_timeout=max_timeout,
            sse_interval=sse_interval,
        )


class DevConfig(ConfigUtil):
    LOGLEVEL = "DEBUG"

    def __init__(self):
        config_file = "config-dev.yaml"
        super().__init__(config_file)

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        self.set_logging_level()
        log_config_path = self.base_dir / "log_config.yaml"
        self.log_config_path = (
            str(log_config_path) if log_config_path.exists() else None
        )


class EdgeConfig(ConfigUtil):
    LOGLEVEL = "INFO"

    def __init__(self):
        config_file = "config-edge.yaml"
        super().__init__(config_file)
        self._init_json_logging()


class StagingConfig(ConfigUtil):
    LOGLEVEL = "INFO"

    def __init__(self):
        config_file = "config-staging.yaml"
        super().__init__(config_file)
        self._init_json_logging()


class ProdConfig(ConfigUtil):
    LOGLEVEL = "WARNING"

    def __init__(self):
        config_file = "config-prod.yaml"
        super().__init__(config_file)
        self._init_json_logging()
