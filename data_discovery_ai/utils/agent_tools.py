#  toolbox contains common tools shared by agents
from typing import Any, Union
import pickle
from data_discovery_ai import logger
from transformers import AutoTokenizer, TFBertModel
import numpy as np
from pathlib import Path


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


def get_text_embedding(text: str) -> np.ndarray:
    """
    Calculates the embedding of a given text using a pre-trained BERT model. This function tokenizes the input text, processes it through a BERT model, and extracts the CLS token embedding, which serves as a representation of the text.
    Input:
        text: str. A piece of textual information. In the context of keyword classification task, this is the abstract of a metadata record.
    Output:
        text_embedding: np.ndarray. A numpy array representing the text embedding as a feature vector. The shape of the output is (768,).
    """
    # https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/bert#transformers.TFBertModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # use in Tensorflow https://huggingface.co/google-bert/bert-base-uncased
    model = TFBertModel.from_pretrained("bert-base-uncased")

    # https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.return_tensors, set as 'tf' to return tensorflow tensor
    inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True)
    outputs = model(inputs)
    text_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    # output as a 1D array, shape (768,)
    return text_embedding.squeeze()
