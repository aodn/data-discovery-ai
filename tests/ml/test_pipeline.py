import sys
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

mlflow_stub = MagicMock()


class DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


mlflow_stub.start_run.return_value = DummyRun()
sys.modules["mlflow"] = mlflow_stub
sys.modules["mlflow.tracking"] = mlflow_stub.tracking

from data_discovery_ai.ml.pipeline import (
    KeywordClassificationPipeline,
    DeliveryClassificationPipeline,
    main,
)
from data_discovery_ai.config.config import (
    KeywordClassificationTrainerConfig,
    DeliveryClassificationTrainerConfig,
)


class TestKeywordClassificationPipeline(unittest.TestCase):
    def setUp(self):
        configUtil = patch("data_discovery_ai.ml.pipeline.ConfigUtil.get_config")
        preprocessor = patch("data_discovery_ai.ml.pipeline.KeywordPreprocessor")
        self.addCleanup(configUtil.stop)
        self.addCleanup(preprocessor.stop)
        self.mock_ConfigUtil = configUtil.start()
        self.mock_Preprocessor = preprocessor.start()

        config_instance = self.mock_ConfigUtil.return_value
        config_instance.base_dir = MagicMock()

        save_p = patch("data_discovery_ai.ml.pipeline.save_to_file")
        load_p = patch("data_discovery_ai.ml.pipeline.load_from_file")
        train_p = patch("data_discovery_ai.ml.pipeline.train_keyword_model")
        self.addCleanup(save_p.stop)
        self.addCleanup(load_p.stop)
        self.addCleanup(train_p.stop)
        self.mock_save = save_p.start()
        self.mock_load = load_p.start()
        self.mock_train = train_p.start()

        config_instance.get_keyword_trainer_config.return_value = (
            KeywordClassificationTrainerConfig(
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
        )
        self.pipeline = KeywordClassificationPipeline()
        self.preprocessor = self.mock_Preprocessor.return_value

    @patch("data_discovery_ai.ml.pipeline.logger")
    def test_is_valid_model_valid(self, mock_logger):
        pipeline = KeywordClassificationPipeline()
        for name in ["development", "Experimental", "PRODUCTION"]:
            self.assertTrue(pipeline.is_valid_model(name))
        mock_logger.error.assert_not_called()

    def test_pipeline_preprocess_start_from_preprocess(self):
        self.preprocessor.fetch_raw_data.return_value = "raw_data"
        self.preprocessor.filter_raw_data.return_value = "preprocessed_data"
        self.preprocessor.calculate_embedding.return_value = (
            "preprocessed_data_with_embedding"
        )
        self.preprocessor.data = SimpleNamespace(labels="labels")

        self.pipeline.pipeline(start_from_preprocess=True, model_name="development")

        # expect to start from preprocessing, including: fetch raw data, preprocess raw data, calculate embeddings, prepare train and test set, and call training model
        self.preprocessor.fetch_raw_data.assert_called_once()
        self.preprocessor.filter_raw_data.assert_called_once_with(raw_data="raw_data")
        self.preprocessor.calculate_embedding.assert_called_once_with(
            ds="preprocessed_data", seperator="|"
        )
        self.preprocessor.prepare_train_test_set.assert_called_once_with(
            raw_data="preprocessed_data_with_embedding"
        )
        self.assertEqual(self.mock_save.call_count, 2)
        self.mock_train.assert_called_once_with("development", self.preprocessor)


class TestDeliveryClassificationPipeline(unittest.TestCase):
    def setUp(self):
        configUtil = patch("data_discovery_ai.ml.pipeline.ConfigUtil")
        preprocessor = patch("data_discovery_ai.ml.pipeline.DeliveryPreprocessor")
        self.addCleanup(configUtil.stop)
        self.addCleanup(preprocessor.stop)
        self.mock_ConfigUtil = configUtil.start()
        self.mock_Preprocessor = preprocessor.start()

        config_instance = self.mock_ConfigUtil.return_value
        config_instance.base_dir = MagicMock()

        save_p = patch("data_discovery_ai.ml.pipeline.save_to_file")
        load_p = patch("data_discovery_ai.ml.pipeline.load_from_file")
        train_p = patch("data_discovery_ai.ml.pipeline.train_delivery_model")
        self.addCleanup(save_p.stop)
        self.addCleanup(load_p.stop)
        self.addCleanup(train_p.stop)
        self.mock_save = save_p.start()
        self.mock_load = load_p.start()
        self.mock_train = train_p.start()

        add_manual_p = patch("data_discovery_ai.ml.pipeline.add_manual_labelled_data")
        self.addCleanup(add_manual_p.stop)
        self.mock_add_manual = add_manual_p.start()
        self.mock_add_manual.return_value = "data_with_manual_labels"

        config_instance.get_delivery_trainer_config.return_value = (
            DeliveryClassificationTrainerConfig(
                test_size=0.2,
                threshold=0.5,
                separator="[SEP]",
                max_depth=5,
                max_leaf_nodes=2,
                max_iter=10,
                k_best=5,
            )
        )
        self.pipeline = DeliveryClassificationPipeline()
        self.preprocessor = self.mock_Preprocessor.return_value

    def test_pipeline_not_start_from_preprocess(self):
        self.mock_load.side_effect = [
            "saved_preprocessed_data_with_embedding",
            "manual_labelled_data",
        ]

        self.pipeline.pipeline(start_from_preprocess=False, model_name="development")

        # expect to start from prepare train and test sets, and call training model
        self.preprocessor.fetch_raw_data.assert_not_called()
        self.preprocessor.filter_raw_data.assert_not_called()
        self.preprocessor.calculate_embedding.assert_not_called()

        self.assertEqual(self.mock_load.call_count, 2)
        self.mock_add_manual.assert_called_once_with(
            "saved_preprocessed_data_with_embedding", "manual_labelled_data"
        )
        self.mock_train.assert_called_once_with("development", self.preprocessor)


class TestMainFunction(unittest.TestCase):
    @patch("data_discovery_ai.ml.pipeline.KeywordClassificationPipeline")
    @patch("data_discovery_ai.ml.pipeline.argparse.ArgumentParser")
    def test_main_keyword(self, mock_ap, mock_kw_pipe):
        args = SimpleNamespace(
            pipeline="keyword", start_from_preprocess=False, model_name="development"
        )
        parser = MagicMock()
        parser.parse_args.return_value = args
        mock_ap.return_value = parser

        pipeline_instance = mock_kw_pipe.return_value
        main()
        parser.parse_args.assert_called_once()
        pipeline_instance.pipeline.assert_called_once_with(False, "development")

    @patch("data_discovery_ai.ml.pipeline.DeliveryClassificationPipeline")
    @patch("data_discovery_ai.ml.pipeline.argparse.ArgumentParser")
    def test_main_delivery(self, mock_ap, mock_del_pipe):
        args = SimpleNamespace(
            pipeline="delivery", start_from_preprocess=True, model_name="experimental"
        )
        parser = MagicMock()
        parser.parse_args.return_value = args
        mock_ap.return_value = parser

        pipeline_instance = mock_del_pipe.return_value
        main()
        parser.parse_args.assert_called_once()
        pipeline_instance.pipeline.assert_called_once_with(True, "experimental")

    @patch("data_discovery_ai.ml.pipeline.logger")
    @patch("data_discovery_ai.ml.pipeline.argparse.ArgumentParser")
    def test_main_invalid(self, mock_ap, mock_logger):
        args = SimpleNamespace(
            pipeline="not_supported_pipeline",
            start_from_preprocess=True,
            model_name="test",
        )
        parser = MagicMock()
        parser.parse_args.return_value = args
        mock_ap.return_value = parser

        main()
        mock_logger.error.assert_called_once_with("Invalid pipeline")


if __name__ == "__main__":
    unittest.main()
