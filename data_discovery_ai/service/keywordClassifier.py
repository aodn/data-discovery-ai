import data_discovery_ai.utils.preprocessor as preprocessor
import data_discovery_ai.model.keywordModel as model
import configparser
from typing import List
from pathlib import Path

from data_discovery_ai.common.constants import KEYWORD_CONFIG

# Base directory where your Poetry project's pyproject.toml is located
BASE_DIR = Path(__file__).resolve().parent.parent

# Construct the full path to the .ini file
config_file_path = BASE_DIR / "common" / KEYWORD_CONFIG
if not config_file_path.exists():
    raise FileNotFoundError(
        f"The configuration file was not found at {config_file_path}"
    )


def keywordClassifier(trained_model, description):
    """
    The keyword classifier service for API use.
    Input:
        trained_model: str. The name of the trained model file (without extension), located in the `data_discovery_ai/resources/` directory. E.g. to load from file `data_discovery_ai/output/pretrainedKeyword4demo.keras`, `traind_model=pretrainedKeyword4demo`.
        description: str. The abstract of a metadata record for predicting the keywords of the dataset.
    Output:
        predicted_keyword: str. the predicted keywords, separate by " | ".
    """

    params = configparser.ConfigParser()
    params.read(config_file_path)

    selected_model = model.load_saved_model(trained_model)
    description_embedding = preprocessor.get_description_embedding(description)
    dimension = description_embedding.shape[0]
    target_X = description_embedding.reshape(1, dimension)
    target_predicted_labels = model.prediction(
        target_X,
        selected_model,
        params.getfloat("keywordModel", "confidence"),
        params.getint("keywordModel", "top_N"),
    )

    labels = preprocessor.load_from_file("labels.pkl")
    prediction = model.get_predicted_keywords(target_predicted_labels, labels).to_list()
    return prediction

