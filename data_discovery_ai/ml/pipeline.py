# ML pipeline, may need to be deployed in cloud environment in the future
import pandas as pd

from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.agent_tools import save_to_file, load_from_file
from data_discovery_ai.ml.preprocessor import KeywordPreprocessor
from data_discovery_ai.config.constants import (
    KEYWORD_FOLDER,
    KEYWORD_SAMPLE_FILE,
    KEYWORD_LABEL_FILE,
    CACHED_RAW_DATA,
)
from data_discovery_ai.ml.keywordModel import train_keyword_model
import argparse
import subprocess
from dotenv import load_dotenv
import os
import time
import socket
import structlog

logger = structlog.get_logger(__name__)


class BasePipeline:
    def __init__(self):
        self.config = ConfigUtil.get_config()
        self.valid_model_name = [
            "development",
            "staging",
            "production",
            "experimental",
            "benchmark",
        ]
        self.preprocessor = None

    def is_valid_model(self, model_name: str) -> bool:
        """
        Validate model name within fixed selections
        Input:
            model_name: str. The file name of the saved model. restricted within these options: ["development", "staging", "production", "experimental", "benchmark"]
        """
        valid_model_name = self.valid_model_name
        if model_name.lower() not in valid_model_name:
            logger.error(
                'Model name invalid! \nAvailable model name: ["development", "staging", "production", "experimental", "benchmark"]'
            )
            return False
        else:
            return True

    def get_raw_data(self, use_cached_raw: bool) -> pd.DataFrame:
        raw_data = None
        cache_path = self.config.base_dir / "resources" / CACHED_RAW_DATA
        if use_cached_raw:
            # use cached raw data from saved file
            try:
                raw_data = load_from_file(cache_path)
                logger.info("Loaded cached raw data from %s", cache_path)
            except FileNotFoundError:
                logger.warning(
                    "Cached raw data not found at %s, fetching from source", cache_path
                )
                raw_data = self.preprocessor.fetch_raw_data()
                save_to_file(raw_data, cache_path)

        else:
            # fetch raw data
            raw_data = self.preprocessor.fetch_raw_data()
            # save as cached file
            save_to_file(raw_data, cache_path)
        return raw_data

    def pipeline(
        self, use_cached_raw: bool, start_from_preprocess: bool, model_name: str
    ) -> None:
        pass

    def start_mlflow(self) -> None:
        """
        Start mlflow server and gateway background, the server log is saved in mlflow_server.log and gateway.log, separately.
        :return:
        """
        load_dotenv()

        mlflow_config = self.config.get_mlflow_config()

        # check if port is in use or not
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", mlflow_config.port)) == 0:
                logger.warning(
                    f"Port {mlflow_config.port} is already in use. view at http://localhost:{mlflow_config.port})"
                )
                return

        logger.info("Exporting OPENAI_API_KEY...")
        logger.info("Starting MLflow Gateway in background...")

        gateway_port = mlflow_config.gateway.split(":")[2]
        subprocess.Popen(
            [
                "mlflow",
                "gateway",
                "start",
                "--config-path",
                "mlflow_config.yaml",
                "--port",
                gateway_port,
            ],
            stdout=open("gateway.log", "w"),
            stderr=subprocess.STDOUT,
        )

        time.sleep(5)

        os.environ["MLFLOW_DEPLOYMENTS_TARGET"] = mlflow_config.gateway

        logger.info(
            "Starting MLflow Server in background, view at http://localhost:{}".format(
                mlflow_config.port
            )
        )
        subprocess.Popen(
            ["mlflow", "server", "--port", str(mlflow_config.port)],
            stdout=open("mlflow_server.log", "w"),
            stderr=subprocess.STDOUT,
        )


class KeywordClassificationPipeline(BasePipeline):
    def __init__(self) -> None:
        """
        Init the pipeline, load parameters from file.
        Input:
            start_from_preprocess: bool. A flag to show whether the data (metadata records) significantly changed. Set as True if data changed, which means sample set needs to be repreprocessed, as well as the model need to be re-trained.
            model_name: str. The model name that saved in a .keras file.
        """
        # extends the BasePipeline class
        super().__init__()
        self.params = self.config.get_keyword_trainer_config()
        self.preprocessor = KeywordPreprocessor()

    def pipeline(
        self, use_cached_raw: bool, start_from_preprocess: bool, model_name: str
    ) -> None:
        """
        The keyword classification model training pipeline.
        Inputs:
            use_cached_raw: bool. If True, load previously fetched raw data from cache. If False, fetch fresh raw data from OGCAPI and (optionally) update the cache.
            start_from_preprocess: bool. If True, run the preprocessing pipeline from raw data and overwrite the cached preprocessed artifacts. If False, load preprocessed data/labels from previously saved files.
            model_name: str. The model name for saving a selected pretrained model.
        """
        executable = self.is_valid_model(model_name)

        if not executable:
            return

        if start_from_preprocess:
            # get raw data
            raw_data = self.get_raw_data(use_cached_raw)
            # preprocess raw data
            filtered_data = self.preprocessor.filter_raw_data(raw_data=raw_data)
            preprocessed_label = self.preprocessor.concept_to_index

            # save textual labels for future use
            save_to_file(
                preprocessed_label,
                self.config.base_dir
                / "resources"
                / KEYWORD_FOLDER
                / KEYWORD_LABEL_FILE,
            )

            # add the embedding column
            preprocessed_data = self.preprocessor.calculate_embedding(
                ds=filtered_data, seperator=self.params.separator
            )

            # save preprocessed data
            save_to_file(
                preprocessed_data,
                self.config.base_dir
                / "resources"
                / KEYWORD_FOLDER
                / KEYWORD_SAMPLE_FILE,
            )

        else:
            preprocessed_data = load_from_file(
                self.config.base_dir
                / "resources"
                / KEYWORD_FOLDER
                / KEYWORD_SAMPLE_FILE
            )

            preprocessed_label = load_from_file(
                self.config.base_dir
                / "resources"
                / KEYWORD_FOLDER
                / KEYWORD_LABEL_FILE,
            )

        if preprocessed_data is None or preprocessed_label is None:
            raise FileNotFoundError("Pretrained model resources not found.")
            # prepare train test sets
        self.preprocessor.prepare_train_test_set(raw_data=preprocessed_data)

        # train model
        train_keyword_model(model_name, self.preprocessor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        help="The ML pipeline to use, options: {'keyword', 'delivery'}",
    )
    parser.add_argument(
        "-r",
        "--use_cached_raw",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use cached raw data instead of fetching fresh data (True/False)",
    )
    parser.add_argument(
        "-s",
        "--start_from_preprocess",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to start from preprocess (True/False)",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="experimental",
        help="The model name want to saved in a .keras file.",
    )

    args = parser.parse_args()
    if args.pipeline == "keyword":
        pipeline = KeywordClassificationPipeline()
        pipeline.start_mlflow()
        pipeline.pipeline(
            args.use_cached_raw, args.start_from_preprocess, args.model_name
        )
    else:
        logger.error("Invalid pipeline")


if __name__ == "__main__":
    main()
