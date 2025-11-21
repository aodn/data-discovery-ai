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
                "concepts": [
                    {
                        "id": "1",
                        "url": "https://vocabs.ardc.edu.au/1",
                        "title": "A",
                        "description": "",
                    }
                ],
                "scheme": "platform",
            }
        ]
        themes_2 = [
            {
                "concepts": [
                    {
                        "id": "2",
                        "url": "https://vocabs.ardc.edu.au/2",
                        "title": "A",
                        "description": "",
                    }
                ],
                "scheme": "theme",
            }
        ]
        df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "title": ["title 1", "title 2"],
                "description": ["abstract 1", "abstract 2"],
                "themes": [themes_1, themes_2],
                "statement": ["statement 1", "statement 2"],
                "status": ["OnGoing", "Completed"],
            }
        )

        out = kp.filter_raw_data(df)

        self.assertListEqual(
            list(out.columns), ["id", "title", "description", "themes"]
        )
        self.assertEqual(out["id"].iloc[0], "1")
        self.assertEqual(out["title"].iloc[0], "title 1")

        # the expected format of keyword is a dict like {"vocab_type": vocab_type, "value": value,"url": url}
        kw0 = out["themes"].iloc[0]
        self.assertIsInstance(kw0, list)
        self.assertEqual(kw0, [0])


class TestDeliveryPreprocessorPostFilter(unittest.TestCase):
    def test_filter_raw_data(self):
        dp = DeliveryPreprocessor()
        themes_1 = [
            {
                "concepts": [
                    {
                        "id": "1",
                        "url": "https://vocabs.ardc.edu.au/1",
                        "title": "A",
                        "description": "",
                    }
                ],
                "scheme": "platform",
            }
        ]
        themes_2 = [
            {
                "concepts": [
                    {
                        "id": "2",
                        "url": "https://vocabs.ardc.edu.au/2",
                        "title": "A",
                        "description": "",
                    }
                ],
                "scheme": "theme",
            }
        ]
        df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "title": ["title real time", "title delayed"],
                "description": ["abstract 1", "abstract 2"],
                "themes": [themes_1, themes_2],
                "statement": ["statement 1", "statement 2"],
                "status": ["onGoing", "Completed"],
            }
        )
        out = dp.filter_raw_data(df)
        self.assertEqual(
            list(out.columns),
            ["id", "title", "description", "statement", "status", "mode"],
        )
        self.assertTrue((out["status"] == "onGoing").all())


if __name__ == "__main__":
    unittest.main()
