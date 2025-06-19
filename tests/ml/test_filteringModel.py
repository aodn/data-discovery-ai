import sys
import unittest
from unittest.mock import MagicMock, patch

# mock mlflow so that don't need running server
mlflow_stub = MagicMock()


class DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


mlflow_stub.start_run.return_value = DummyRun()
sys.modules["mlflow"] = mlflow_stub
sys.modules["mlflow.tracking"] = mlflow_stub.tracking

import data_discovery_ai.ml.filteringModel as filteringModel


class TestFilteringModel(unittest.TestCase):
    def test_get_predicted_class_name(self):
        self.assertEqual(filteringModel.get_predicted_class_name(0), "Real-Time")
        self.assertEqual(filteringModel.get_predicted_class_name(1), "Delayed")
        self.assertEqual(filteringModel.get_predicted_class_name(2), "Other")
        self.assertIsNone(filteringModel.get_predicted_class_name(3))

    @patch("data_discovery_ai.ml.filteringModel.load_from_file")
    def test_load_saved_model(self, mock_load):
        mock_model = MagicMock()
        mock_load.side_effect = [mock_model]

        model = filteringModel.load_saved_model("test")

        self.assertEqual(mock_load.call_count, 1)
        self.assertIs(model, mock_model)


if __name__ == "__main__":
    unittest.main()
