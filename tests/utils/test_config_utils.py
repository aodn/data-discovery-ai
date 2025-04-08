# unit test for config_utils.py
import unittest
from unittest.mock import patch, MagicMock

from data_discovery_ai.utils.config_utils import ConfigUtil
from data_discovery_ai.common.constants import MODEL_CONFIG, ELASTICSEARCH_CONFIG
import configparser
from pathlib import Path


class TestConfigUtil(unittest.TestCase):
    def setUp(self):
        self.config = ConfigUtil()

    # test if configuration files exits -> return True
    @patch("data_discovery_ai.utils.config_utils.Path.exists", return_value=True)
    @patch("data_discovery_ai.utils.config_utils.configparser.ConfigParser")
    def test_load_model_config_success(self, mock_parser_class, mock_exists):
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser

        result = self.config.load_model_config()
        config_path = self.config.base_dif / "common" / MODEL_CONFIG

        mock_exists.assert_called_once()
        mock_parser.read.assert_called_once_with(config_path)
        self.assertEqual(result, mock_parser)

    # test if configuration files does not exits -> raise FileNotFoundError
    @patch("data_discovery_ai.utils.config_utils.Path.exists", return_value=False)
    def test_load_config_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError) as context:
            self.config.load_es_config()

        expected_path = self.config.base_dif / "common" / ELASTICSEARCH_CONFIG
        self.assertIn(str(expected_path), str(context.exception))


if __name__ == "__main__":
    unittest.main()
