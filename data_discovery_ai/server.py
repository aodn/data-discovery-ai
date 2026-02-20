from pathlib import Path
import uvicorn
from fastapi import FastAPI
from data_discovery_ai.utils.es_connector import create_es_index
from data_discovery_ai.core.routes import router as api_router
from transformers import AutoTokenizer, TFBertModel, TFAutoModelForSequenceClassification
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import structlog

from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.enum.agent_enums import HuggingfaceModel

logger = structlog.get_logger(__name__)


def load_embedding_tokenizer_model():
    # https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/bert#transformers.TFBertModel
    embedding_tokenizer = AutoTokenizer.from_pretrained(HuggingfaceModel.EMBEDDING_MODEL_NAME.value)
    # use in Tensorflow https://huggingface.co/google-bert/bert-base-uncased
    embedding_model = TFBertModel.from_pretrained(HuggingfaceModel.EMBEDDING_MODEL_NAME.value)

    return embedding_tokenizer, embedding_model

def load_nli_tokenizer_model():
    nli_tokenizer = AutoTokenizer.from_pretrained(HuggingfaceModel.NLI_MODEL_NAME.value)
    nli_model = TFAutoModelForSequenceClassification.from_pretrained(HuggingfaceModel.NLI_MODEL_NAME.value)
    return nli_tokenizer, nli_model


def load_llm_client():
    load_dotenv()
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm_client = AsyncOpenAI(api_key=openai_api_key)
        return llm_client
    except Exception:
        logger.error("Failed to start server: OpenAI API key required")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_tokenizer, embedding_model = load_embedding_tokenizer_model()
    app.state.tokenizer = embedding_tokenizer
    app.state.embedding_model = embedding_model

    nli_tokenizer, nli_model = load_nli_tokenizer_model()
    app.state.nli_tokenizer = nli_tokenizer
    app.state.nli_model = nli_model

    # create Elasticsearch index
    client, index = create_es_index()
    app.state.client = client
    app.state.index = index

    # create OpenAI client
    app.state.llm_client = load_llm_client()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)

if __name__ == "__main__":
    config = ConfigUtil.get_config()
    log_config_path = config.log_config_path
    app_config = config.get_application_config()
    uvicorn.run(
        "data_discovery_ai.server:app",
        host="0.0.0.0",
        port=app_config.port,
        reload=app_config.reload,
        log_config=log_config_path,
        timeout_keep_alive=900,
    )
