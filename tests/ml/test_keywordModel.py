import sys
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

mlflow_stub = MagicMock()


class DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


mlflow_stub.start_run.return_value = DummyRun()
sys.modules["mlflow"] = mlflow_stub
sys.modules["mlflow.tracking"] = mlflow_stub.tracking

import data_discovery_ai.ml.keywordModel as keywordModel


class TestKeywordModel(unittest.TestCase):
    def test_replace_with_column_name(self):
        test_row = pd.Series([1, 0, 1])
        test_rows = {0: "vocab 1", 1: "vocab 2", 2: "vocab 3"}
        result = keywordModel.replace_with_column_names(test_row, test_rows)
        self.assertEqual(result, ["vocab 1", "vocab 3"])

    @patch("data_discovery_ai.ml.keywordModel.load_model")
    @patch("data_discovery_ai.ml.keywordModel.logger")
    def test_load_model_succeed(self, mock_logger, mock_load_model):
        saved_model = MagicMock()
        mock_load_model.return_value = saved_model

        result = keywordModel.load_saved_model("test")

        mock_load_model.assert_called_once()
        args, kwargs = mock_load_model.call_args
        self.assertIsInstance(args[0], Path)
        self.assertEqual(kwargs.get("compile"), False)

        self.assertIs(result, saved_model)
        mock_logger.error.assert_not_called()
