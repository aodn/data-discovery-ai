# The agent model for executing the keyword classification task
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Iterable
from keras.models import load_model  # type: ignore

from data_discovery_ai import logger
from data_discovery_ai.agents.baseAgent import BaseAgent
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.utils.agent_tools import get_text_embedding, load_from_file
from data_discovery_ai.config.constants import KEYWORD_FOLDER, KEYWORD_LABEL_FILE
from data_discovery_ai.ml.preprocessor import Concept, ConceptTheme


def replace_with_column_names(
    row: Iterable[Any], column_names: Dict[int, Any]
) -> List[Any]:
    """
    Transform a row of binary values and returns a list of column names for which the value in the row is 1.
    Input:
        row: Iterable[Any]. A row of binary values indicating presence (1) or absence (0) of each label.
        column_names: Dict[int, Any]. The anonymous predefined label set.
    Output:
        List[Any]: The predicted keywords as a list of concepts.
    """
    return [column_names[i] for i, value in enumerate(row) if value == 1]


def get_predicted_keywords(prediction: np.ndarray, labels: Dict):
    """
    Convert binary predictions to textual keywords.
    Input:
        prediction: np.ndarray. The predicted binary matrix.
        labels: Dict. The predefined keywords.
    Output:
        predicted_keywords: pd.Series. The predicted keywords for the given targets.
    """
    target_predicted = pd.DataFrame(prediction, columns=list(labels.keys()))
    predicted_keywords = target_predicted.apply(
        lambda row: replace_with_column_names(row, labels), axis=1
    )
    if len(predicted_keywords) == 1:
        predicted_keywords = [item for item in predicted_keywords[0]]
    return predicted_keywords


def reformat_response(response: Dict) -> Dict:
    theme_map = {}

    for item in response.get("themes", []):
        theme_info = item.get("theme", {})
        theme_key = (
            theme_info.get("title"),
            theme_info.get("scheme"),
            theme_info.get("description", ""),
        )

        if theme_key not in theme_map:
            theme = ConceptTheme(
                title=theme_key[0],
                scheme=theme_key[1],
                description=theme_key[2],
            )
            theme.set_as_ai_prediction("ai:description", "")
            theme_map[theme_key] = theme
        else:
            theme = theme_map[theme_key]

        concept = Concept(value=item["id"], url=item["url"])
        concept.set_theme(theme)
        theme.add_concept(concept)

    formatted = {
        "themes": [
            {
                "concepts": [c.to_json() for c in theme.concepts],
                **theme.to_json(),
                "ai:description": "This is the prediction provided by AI model",
            }
            for theme in theme_map.values()
        ]
    }

    return formatted


class KeywordClassificationAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.type = "keyword_classification"
        self.config = ConfigUtil()
        self.model_config = self.config.get_keyword_classification_config()

        self.pretrained_model_name = self.model_config.pretrained_model
        self.supervisor = None

    def set_supervisor(self, supervisor):
        self.supervisor = supervisor

    def set_required_fields(self, required_fields) -> None:
        return super().set_required_fields(required_fields)

    def is_valid_request(self, request: Dict[str, str]) -> bool:
        return super().is_valid_request(request)

    def make_decision(self, request) -> bool:
        # the agent will always take action if the request is valid, as the decision is made from es-indexer
        return self.is_valid_request(request)

    def execute(self, request: dict) -> None:
        """
        Execute the action module of the Keyword Classification Agent. The action is to classify the keywords based on the provided request.
        The agent perceives the request, and make decision based on the received request. If it decides to take action, it will call the LLM module to classify the keywords and set self response as the classified keywords.
        Otherwise, it will set self.response as an empty list.
        Input:
            request (dict): The request format.
        """
        flag = self.make_decision(request)
        if not flag:
            self.response = {self.model_config.response_key: []}
        else:
            title = request["title"]
            abstract = request["abstract"]
            prediction = self.take_action(title, abstract)
            self.response = {self.model_config.response_key: prediction}
            self.response = reformat_response(self.response)
        logger.info(f"{self.type} agent finished, it responses: \n {self.response}")

    def take_action(self, title, abstract) -> List[Any]:
        """
        The action module of the Keyword Classification Agent. Its task is to classify the keywords based on the provided title and abstract.
        """
        # calculate the embedding of the title and abstract
        request_text = title + self.model_config.separator + abstract
        tokenizer = self.supervisor.tokenizer
        embedding_model = self.supervisor.embedding_model
        text_embedding = get_text_embedding(request_text, tokenizer, embedding_model)

        # set up model parameters
        confidence = self.model_config.confidence
        top_N = self.model_config.top_N

        # load the pretrained model
        pretrained_model = self.load_saved_model()
        selective_labels = self.load_keyword_labels()
        if pretrained_model and selective_labels:
            dimension = text_embedding.shape[0]
            target_X = text_embedding.reshape(1, dimension)
            predictions = pretrained_model.predict(target_X)
            predicted_labels = (predictions > confidence).astype(int)

            for i in range(predicted_labels.shape[0]):
                if predicted_labels[i].sum() == 0:
                    top_indices = np.argsort(predictions[i])[-top_N:]
                    predicted_labels[i][top_indices] = 1

            predicted_keywords = get_predicted_keywords(
                predicted_labels, selective_labels
            )

            logger.info(f"Keyword is being predicted by {self.type} agent")
            return predicted_keywords
        else:
            return []

    def load_saved_model(self) -> Optional[load_model]:
        pretrained_model_name = self.model_config.pretrained_model
        model_file_path = (
            self.config.base_dir / "resources" / KEYWORD_FOLDER / pretrained_model_name
        ).with_suffix(".keras")
        try:
            saved_model = load_model(model_file_path, compile=False)
            return saved_model
        except OSError:
            logger.error(
                f"Failed to load selected model {pretrained_model_name} from folder {model_file_path}"
            )
            return None

    def load_keyword_labels(self) -> Dict[str, Any]:
        """
        Load the keyword labels from the file.
        Output:
            labels: Dict. The predefined keywords.
        """
        labels_file_path = (
            self.config.base_dir / "resources" / KEYWORD_FOLDER / KEYWORD_LABEL_FILE
        )
        labels = load_from_file(labels_file_path)
        return labels


if __name__ == "__main__":
    agent = KeywordClassificationAgent()
    request = {
        "title": "IMOS - Satellite Remote Sensing - MODIS - 01 day - Net Primary Productivity (GSM model and Eppley-VGPM algorithm)",
        "abstract": """
        The Aqua satellite platform carries a MODIS sensor that observes sunlight reflected from within the ocean surface layer at multiple wavelengths. These multi-spectral measurements are used to infer the concentration of chlorophyll-a (Chl-a), most typically due to phytoplankton, present in the water. Modelling is then used to compute an estimate of the Net Primary Productivity (NPP).

The model used is based on the standard vertically generalised production model (VGPM).  The VGPM is a "chlorophyll-based" model that estimates net primary production from chlorophyll using a temperature-dependent description of chlorophyll-specific        photosynthetic efficiency. For the VGPM, net primary production is a function of chlorophyll, available light, and the photosynthetic efficiency. The only difference between the Standard VGPM and the Eppley-VGPM is the temperature-dependent description of photosynthetic efficiencies, with the Eppley approach using an exponential function to account for variation in photosynthetic efficiencies due to photoacclimation. The similarity between these models (VGPM vs E_VGPM) is described more extensively in a paper by Elena-Carr et al (2006.). VGPM is a Depth Integrated Model(DIM) described by Behrenfeld and Falkowski (1997), with modification due to Eppley (1972) as implemented by Antoine and Morel (1996).

There are multiple retrieval algorithms for estimating Chl-a. These data (eppley_npp_chl_gsm) use the Garver-Siegel-Maritorena (GSM) method implemented in the SeaDAS processing software l2gen and described in “Chapter 11, and references therein, of IOCCG Report 5, 2006, (http://ioccg.org/wp-content/uploads/2015/10/ioccg-report-05.pdf). This product is recommended for use in the VGPM model. The product Eppley_NPP_chl_gsm is experimental, in the sense that it is challenging to validate due to the scarcity of insitu measurments. It should only be used with caution informed by understanding of the algorithms.

Production of contemporary MODIS data ceased at 2024/12/26. Users requiring similar Ocean Colour Radiometry data beyond 2024 should consider the VIIRS instruments on the SNPP and NOAA20+ series of satellites, also provided by IMOS.
        """,
    }

    agent.execute(request)
