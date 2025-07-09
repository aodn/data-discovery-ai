from pathlib import Path
import uvicorn
from fastapi import FastAPI
from data_discovery_ai.utils.es_connector import create_es_index
from data_discovery_ai.core.routes import router as api_router
from transformers import AutoTokenizer, TFBertModel
from contextlib import asynccontextmanager

from data_discovery_ai.config.config import ConfigUtil


def load_tokenizer_model():
    # https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/bert#transformers.TFBertModel
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    # use in Tensorflow https://huggingface.co/google-bert/bert-base-uncased
    embedding_model = TFBertModel.from_pretrained("google-bert/bert-base-uncased")

    return tokenizer, embedding_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer, embedding_model = load_tokenizer_model()
    app.state.tokenizer = tokenizer
    app.state.embedding_model = embedding_model

    # create Elasticsearch index
    client, index = create_es_index()
    app.state.client = client
    app.state.index = index
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)

if __name__ == "__main__":
    log_config_path = str(Path(__file__).parent.parent / "log_config.yaml")

    app_config = ConfigUtil.get_config().get_application_config()
    uvicorn.run(
        "data_discovery_ai.server:app",
        host="0.0.0.0",
        port=app_config.port,
        reload=app_config.reload,
        log_config=log_config_path,
        timeout_keep_alive=900,
    )
