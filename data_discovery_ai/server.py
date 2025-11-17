from pathlib import Path
import uvicorn
from fastapi import FastAPI
from data_discovery_ai.utils.es_connector import create_es_index
from data_discovery_ai.core.routes import router as api_router
from transformers import AutoTokenizer, TFBertModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import structlog

from data_discovery_ai.config.config import ConfigUtil

logger = structlog.get_logger(__name__)


def load_tokenizer_model():
    # https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/bert#transformers.TFBertModel
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    # use in Tensorflow https://huggingface.co/google-bert/bert-base-uncased
    embedding_model = TFBertModel.from_pretrained("google-bert/bert-base-uncased")

    return tokenizer, embedding_model


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
    tokenizer, embedding_model = load_tokenizer_model()
    app.state.tokenizer = tokenizer
    app.state.embedding_model = embedding_model

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
    log_config_path = config.get_log_config_path()
    app_config = config.get_application_config()
    uvicorn.run(
        "data_discovery_ai.server:app",
        host="0.0.0.0",
        port=app_config.port,
        reload=app_config.reload,
        log_config=log_config_path,
        timeout_keep_alive=900,
    )
