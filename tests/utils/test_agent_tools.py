# unit test for agent_tools.py
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from data_discovery_ai.utils.agent_tools import (
    save_to_file,
    load_from_file,
    get_text_embedding,
)


class TestAgentTools(unittest.TestCase):
    def setUp(self):
        self.test_obj = {"key": "value"}

    @patch("data_discovery_ai.utils.agent_tools.logger")
    @patch("pickle.dump")
    @patch("builtins.open", new_callable=MagicMock)
    def test_save_to_file(self, mock_open, mock_pickle_dump, mock_logger):
        mock_open.return_value.__enter__.return_value = MagicMock()
        save_to_file(self.test_obj, "test_path.pkl")
        mock_open.assert_called_once_with("test_path.pkl", "wb")
        mock_pickle_dump.assert_called_once_with(
            self.test_obj, mock_open.return_value.__enter__()
        )
        mock_logger.info.assert_called_once_with("Saved to test_path.pkl")

    @patch("data_discovery_ai.utils.agent_tools.logger")
    @patch("pickle.load")
    @patch("builtins.open", new_callable=MagicMock)
    def test_load_from_file(self, mock_open, mock_pickle_load, mock_logger):
        mock_open.return_value.__enter__.return_value = MagicMock()
        mock_pickle_load.return_value = self.test_obj
        result = load_from_file("test_path.pkl")
        mock_open.assert_called_once_with("test_path.pkl", "rb")
        mock_pickle_load.assert_called_once_with(mock_open.return_value.__enter__())
        mock_logger.info.assert_called_once_with("Load from test_path.pkl")
        self.assertEqual(result, self.test_obj)

    @patch("data_discovery_ai.utils.agent_tools.logger")
    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    def test_load_from_file_not_exist(self, mock_open, mock_logger):
        result = load_from_file("not_exist_path.pkl")

        mock_open.assert_called_once_with("not_exist_path.pkl", "rb")
        mock_logger.error.assert_called_once()
        args, _ = mock_logger.error.call_args
        self.assertIsInstance(args[0], FileNotFoundError)
        self.assertEqual(str(args[0]), "File not found")
        self.assertIsNone(result)

    @patch("data_discovery_ai.utils.agent_tools.tokenizer")
    @patch("data_discovery_ai.utils.agent_tools.model")
    def test_get_text_embedding(self, mock_embedding_model, mock_tokenizer_model):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[101]], "attention_mask": [[1]]}
        mock_tokenizer_model.return_value = mock_tokenizer

        # the expected shape should be (768,), assume they all value 1
        mock_output = np.ones((1, 768))
        mock_model_output = MagicMock()
        mock_model_output.last_hidden_state = MagicMock()
        mock_model_output.last_hidden_state.__getitem__.return_value.numpy.return_value = (
            mock_output
        )

        mock_embedding_model.return_value = mock_model_output

        embedding = get_text_embedding("This is a test text")
        self.assertEqual(embedding.shape, (768,))
        self.assertTrue((embedding == 1.0).all())


if __name__ == "__main__":
    unittest.main()
