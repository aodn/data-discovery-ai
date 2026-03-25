#  toolbox contains common tools shared by agents
from typing import Any, Union
import pickle
import structlog
import numpy as np
from pathlib import Path
import json

logger = structlog.get_logger(__name__)


def save_to_file(obj: Any, full_path: Union[str, Path]) -> None:
    """
    Saves an object to a file using pickle serialization in the specific file path
    """
    try:
        with open(full_path, "wb") as file:
            # noinspection PyTypeChecker
            pickle.dump(obj, file)
            logger.info(f"Saved to {full_path}")
    except Exception as e:
        logger.error(e)


def load_from_file(full_path: Union[str, Path]) -> Any | None:
    """
    Loads an object from a file in the input folder using pickle deserialization.
    """
    try:
        with open(full_path, "rb") as file:
            obj = pickle.load(file)
            logger.info(f"Load from {full_path}")
        return obj
    except Exception as e:
        logger.error(e)
        return None


def get_text_embedding(
    text: str, tokenizer: Any, embedding_model: Any, max_length: int = 512
) -> np.ndarray | None:
    """
    Calculates the embedding of a given text using a pre-trained BERT model. This function tokenizes the input text, processes it through a BERT model, and extracts the CLS token embedding, which serves as a representation of the text.
    Input:
        text: str. A piece of textual information. In the context of keyword classification task, this is the abstract of a metadata record.
    Output:
        text_embedding: np.ndarray. A numpy array representing the text embedding as a feature vector. The shape of the output is (768,).
    """
    if not tokenizer or not embedding_model:
        return None
    else:
        # https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.return_tensors, set as 'tf' to return tensorflow tensor
        inputs = tokenizer(
            text, return_tensors="tf", max_length=max_length, truncation=True
        )
        outputs = embedding_model(inputs)
        text_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        # output as a 1D array, shape (768,)
        return text_embedding.squeeze()


def parse_combined_title(combined_title: str) -> tuple[str | None, str | None]:
    """
    Helper function to parse combined text in link.title field, which is a combination of link title and description in the json format {"title": "My Title", "description": "My Description"}.
    The description field must exist in the JSON (even if empty) to parse the title. Otherwise, return the original text.

    :param combined_title: str: the combined json string in link.title field
    :return: tuple[str | None, str | None]. The parsed title and description in string. description can be None if it's an empty string.
    """
    # return None, None for both title and description if the combined text is None or empty
    if combined_title is None or combined_title.strip() == "":
        return None, None

    # Try to parse as JSON
    try:
        data = json.loads(combined_title)

        if not isinstance(data, dict):
            return combined_title.strip(), None

        # Check if description field exists (key must be present)
        if "description" not in data:
            # No description field, return original text
            return combined_title.strip(), None

        # Parse title
        title = data.get("title")
        if isinstance(title, str):
            title = title.strip()
            title = title if title else None
        else:
            title = None

        # Parse description
        description = data.get("description")
        if isinstance(description, str):
            description = description.strip()
            description = description if description else None
        else:
            description = None

        return title, description

    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, return original text as title with None description
        return combined_title.strip(), None
