from fastapi import FastAPI
from data_discovery_ai.core.routes import router as api_router

from transformers import AutoTokenizer, TFBertModel
from contextlib import asynccontextmanager


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
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)
