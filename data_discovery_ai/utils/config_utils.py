from pathlib import Path
from data_discovery_ai.common.constants import MODEL_CONFIG, ELASTICSEARCH_CONFIG
import configparser


class ConfigUtil:
    def __init__(self) -> None:
        self.base_dif = Path(__file__).resolve().parent.parent

    def _load_config(self, file_name: str) -> configparser.ConfigParser:
        """
        The abstract method to load a configuration file.
        """
        config_file_path = self.base_dif / "common" / file_name
        if not config_file_path.exists():
            raise FileNotFoundError(
                f"The configuration file was not found at {config_file_path}"
            )

        config = configparser.ConfigParser()
        config.read(config_file_path)
        return config

    def load_model_config(self) -> configparser.ConfigParser:
        """
        The util method for load parameters for ML models from a configuration file, which saved as data_discovery_ai/common/classification_parameters.ini
        """
        return self._load_config(MODEL_CONFIG)

    def load_es_config(self) -> configparser.ConfigParser:
        """
        The util method for load Elasticsearch configurations from a file, which saved as data_discovery_ai/common/esManager.ini
        """
        return self._load_config(ELASTICSEARCH_CONFIG)
