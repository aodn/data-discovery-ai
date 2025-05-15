# ML pipeline, may need to be deployed in cloud environment in the future
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.agent_tools import save_to_file, load_from_file
from data_discovery_ai.ml.preprocessor import KeywordPreprocessor, DeliveryPreprocessor
from data_discovery_ai.config.constants import (
    KEYWORD_FOLDER,
    KEYWORD_SAMPLE_FILE,
    KEYWORD_LABEL_FILE,
    FILTER_FOLDER,
    FILTER_PREPROCESSED_FILE
)
from data_discovery_ai.ml.keywordModel import train_keyword_model
from data_discovery_ai.ml.filteringModel import train_delivery_model
from data_discovery_ai import logger
import argparse

class BasePipeline:
    def __init__(self):
        self.config = ConfigUtil()
        self.valid_model_name = [
            "development",
            "staging",
            "production",
            "experimental",
            "benchmark",
        ]

    def is_valid_model(self, model_name: str) -> bool:
        """
        Validate model name within fixed selections
        Input:
            model_name: str. The file name of the saved model. restricted within these options: ["development", "staging", "production", "experimental", "benchmark"]
        """
        valid_model_name = self.valid_model_name
        if model_name.lower() not in valid_model_name:
            logger.error(
                'Available model name: ["development", "staging", "production", "experimental", "benchmark"]'
            )
            return False
        else:
            return True

    def pipeline(self, start_from_preprocess: bool, model_name: str) -> None:
        pass


class KeywordClassifierPipeline(BasePipeline):
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

    def pipeline(self, start_from_preprocess: bool, model_name: str) -> None:
        """
        The keyword classification model training pipeline.
        Inputs:
            start_from_preprocess: bool. The indicator to call the data preprocessing module or not.
            model_name: str. The model name for saving a selected pretrained model.
        """
        if start_from_preprocess:
            # fetch raw data
            raw_data = self.preprocessor.fetch_raw_data()

            # # for test only because it has more data
            # raw_data = load_from_file(
            #     self.config.base_dir / "resources" / "raw_data.pkl"
            # )

            # preprocess raw data
            filtered_data = self.preprocessor.filter_raw_data(raw_data=raw_data)

            # add the embedding column
            preprocessed_data = self.preprocessor.calculate_embedding(
                ds=filtered_data, seperator=self.params["separator"]
            )
        else:
            preprocessed_data = load_from_file(
                self.config.base_dir
                / "resources"
                / KEYWORD_FOLDER
                / KEYWORD_SAMPLE_FILE
            )

        # prepare train test sets
        self.preprocessor.prepare_train_test_set(raw_data=preprocessed_data)

        # save preprocessed data for future use
        save_to_file(
            self.preprocessor.data.labels,
            self.config.base_dir / "resources" / KEYWORD_FOLDER / KEYWORD_LABEL_FILE,
        )
        save_to_file(
            preprocessed_data,
            self.config.base_dir / "resources" / KEYWORD_FOLDER / KEYWORD_SAMPLE_FILE,
        )

        # train model
        train_keyword_model(model_name, self.preprocessor)


#
class DeliveryClassificationPipeline(BasePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.params = self.config.get_delivery_trainer_config()
        self.preprocessor = DeliveryPreprocessor()

    def pipeline(self, start_from_preprocess: bool, model_name: str) -> None:
        if start_from_preprocess:
            raw_data = self.preprocessor.fetch_raw_data()
            filtered_data = self.preprocessor.filter_raw_data(raw_data=raw_data)
            preprocessed_data = self.preprocessor.calculate_embedding(
                ds=filtered_data, seperator=self.params["separator"]
            )
            save_to_file(preprocessed_data,
                         self.config.base_dir / "resources" / FILTER_FOLDER / FILTER_PREPROCESSED_FILE)
        else:
            preprocessed_data = load_from_file(
                self.config.base_dir / "resources" / FILTER_FOLDER / FILTER_PREPROCESSED_FILE
            )
        self.preprocessor.prepare_train_test_set(preprocessed_data)

        train_delivery_model(model_name, self.preprocessor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        help="The ML pipeline to use, options: {'keyword', 'delivery'}",
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
        pipeline = KeywordClassifierPipeline()
        pipeline.pipeline(args.start_from_preprocess, args.model_name)
    elif args.pipeline == "delivery":
        pipeline = DeliveryClassificationPipeline()
        pipeline.pipeline(args.start_from_preprocess, args.model_name)
    else:
        logger.error("Invalid pipeline")


if __name__ == "__main__":
    main()
