# unit test for preprocessor.py
import unittest
from unittest.mock import patch
import pandas as pd

from data_discovery_ai.config.config import (
    ConfigUtil,
    KeywordClassificationTrainerConfig,
)
from data_discovery_ai.ml.preprocessor import KeywordPreprocessor, DeliveryPreprocessor


class TestKeywordPreprocessorFilter(unittest.TestCase):
    @patch.object(ConfigUtil, "get_keyword_trainer_config")
    def test_filter_raw_data(self, mock_get_conf):
        mock_get_conf.return_value = KeywordClassificationTrainerConfig(
            vocabs=["A", "B"],
            test_size=0.2,
            n_splits=3,
            dropout=0.1,
            learning_rate=1e-3,
            fl_gamma=2.0,
            fl_alpha=0.25,
            epoch=1,
            batch_size=16,
            early_stopping_patience=2,
            reduce_lr_patience=1,
            validation_split=0.1,
            rare_label_threshold=5,
            separator="|",
        )
        kp = KeywordPreprocessor()

        themes_1 = [
            {
                "concepts": [{"id": "1", "url": "https://vocabs.ardc.edu.au/1"}],
                "scheme": "",
                "description": "",
                "title": "A",
            }
        ]
        themes_2 = [
            {
                "concepts": [{"id": "2", "url": "https://vocabs.ardc.edu.au/2"}],
                "scheme": "",
                "description": "",
                "title": "B",
            }
        ]
        df = pd.DataFrame(
            {
                "_id": ["1", "2"],
                "_source.title": ["title 1", "title 2"],
                "_source.description": ["abstract 1", "abstract 2"],
                "_source.themes": [themes_1, themes_2],
            }
        )

        out = kp.filter_raw_data(df)

        self.assertListEqual(list(out.columns), ["id", "title", "abstract", "keywords"])
        self.assertEqual(out["id"].iloc[0], "1")
        self.assertEqual(out["abstract"].iloc[0], "abstract 1")

        # the expected format of keyword is a dict like {"vocab_type": vocab_type, "value": value,"url": url}
        kw0 = out["keywords"].iloc[0]
        self.assertIsInstance(kw0, list)
        self.assertEqual(kw0, [0])


class TestDeliveryPreprocessorPostFilter(unittest.TestCase):
    def test_filter_raw_data(self):
        dp = DeliveryPreprocessor()
        df = pd.DataFrame(
            {
                "_id": ["1", "2", "3"],
                "_source.title": [
                    "Real time data",
                    "Delayed response",
                    "completed data",
                ],
                "_source.description": [
                    "test real time abstract",
                    "test delayed abstract",
                    "test completed abstract",
                ],
                "_source.summaries.statement": ["a", "b", None],
                "_source.summaries.status": ["onGoing", "onGoing", "completed"],
            }
        )
        out = dp.filter_raw_data(df)
        self.assertTrue(
            list(out.columns), ["id", "title", "abstract", "lineage", "status", "mode"]
        )
        self.assertTrue((out["status"] == "onGoing").all())
        self.assertIn("mode", out.columns)
        self.assertSetEqual(set(out["mode"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
