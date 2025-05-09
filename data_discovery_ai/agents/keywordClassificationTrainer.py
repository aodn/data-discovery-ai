# task agent for training keyword classification model
from typing import Dict, List, Any
import tempfile

import pandas as pd

from data_discovery_ai import logger
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.utils.agent_tools import save_to_file, load_from_file
from data_discovery_ai.ml.preprocessor import KeywordPreprocessor
from data_discovery_ai.config.constants import KEYWORD_FOLDER


class KeywordClassificationTrainer(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "keyword_classification_trainer"
        self.config = ConfigUtil()
        self.model_config = self.config.get_keyword_trainer_config()
        self.valid_model_name = [
            "development",
            "staging",
            "production",
            "experimental",
            "benchmark",
        ]
        self.required_fields = ["start_from_preprocess", "model_name"]
        self.preprocessor = KeywordPreprocessor()

    def set_required_fields(self, required_fields) -> None:
        return super().set_required_fields(required_fields)

    def is_valid_request(self, request: Dict[str, str]) -> bool:
        return super().is_valid_request(request)

    def make_decision(self, request) -> bool:
        # only check if the parameter "model_name" is in the selective options
        return self.is_valid_request(request) and (
            request["model_name"] in self.valid_model_name
        )

    def take_action(self, start_from_preprocess: bool, model_name: str) -> None:
        if start_from_preprocess:
            # fetch raw data
            # raw_data = self.preprocessor.fetch_raw_data()

            # for test only because it has more data
            raw_data = load_from_file(
                self.config.base_dir / "resources" / "raw_data.pkl"
            )

            # preprocess raw data
            filtered_data = self.preprocessor.filter_raw_data(raw_data=raw_data)
            # add the embedding column
            preprocessed_data = self.preprocessor.calculate_embedding(
                ds=filtered_data, seperator=self.model_config["separator"]
            )

    def execute(self, request):
        if not self.is_valid_request(request):
            self.response = {"response": "Invalid request received"}
        else:
            start_from_preprocess = request.get("start_from_preprocess", False)
            model_name = request["model_name"]
            self.take_action(
                start_from_preprocess=start_from_preprocess, model_name=model_name
            )

            self.response = {"response": "Keyword classification trainer finished"}


if __name__ == "__main__":
    keywordClassificationTrainer = KeywordClassificationTrainer()
    trainer_request = {"start_from_preprocess": True, "model_name": "development"}
    keywordClassificationTrainer.execute(trainer_request)
    print(keywordClassificationTrainer.response)
