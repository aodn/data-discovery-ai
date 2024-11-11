from pathlib import Path
from data_discovery_ai.common.constants import KEYWORD_CONFIG, ELASTICSEARCH_CONFIG
import configparser


class ConfigUtil:
    def __init__(self) -> None:
        self.base_dif = Path(__file__).resolve().parent.parent

    def load_keyword_config(self) -> configparser.ConfigParser:
        """
        The util method for load parameters for from a configuration file, which saved as data_discovery_ai/common/keyword_classification_parameters.ini
        """
        config_file_path = self.base_dif / "common" / KEYWORD_CONFIG
        if not config_file_path.exists():
            raise FileNotFoundError(
                f"The configuration file was not found at {config_file_path}"
            )
        keyword_config = configparser.ConfigParser()
        keyword_config.read(config_file_path)
        return keyword_config

    def load_es_config(self) -> configparser.ConfigParser:
        """
        The util method for load Elasticsearch configurations from a file, which saved as data_discovery_ai/common/esManager.ini
        """
        elasticsearch_config_file_path = self.base_dif / "common" / ELASTICSEARCH_CONFIG
        if not elasticsearch_config_file_path.exists():
            raise FileNotFoundError(
                f"The configuration file was not found at {elasticsearch_config_file_path}"
            )
        esConfig = configparser.ConfigParser()
        esConfig.read(elasticsearch_config_file_path)
        return esConfig
