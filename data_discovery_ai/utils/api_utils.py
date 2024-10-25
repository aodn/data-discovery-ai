import os

from dotenv import load_dotenv
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from data_discovery_ai.common.constants import API_KEY_NAME, AVAILABLE_MODELS

# Load environment variables from .env file
load_dotenv()

# Load the API key from environment variables
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name=API_KEY_NAME)


async def api_key_auth(x_api_key: str = Security(api_key_header)):
    if x_api_key == API_KEY:
        return x_api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )


def validate_model_name(selected_model: str):
    if selected_model.lower() in AVAILABLE_MODELS:
        return True
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid selected model name"
    )
