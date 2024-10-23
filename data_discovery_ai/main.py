from fastapi import FastAPI, HTTPException, Depends, Security, APIRouter, status
from fastapi.security import APIKeyHeader
from data_discovery_ai.common.constants import API_PREFIX, API_KEY_NAME
from dotenv import load_dotenv
import os

router = APIRouter(prefix=API_PREFIX)


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

@router.get("/hello", dependencies=[Depends(api_key_auth)])
async def hello():
    return {"content": "Hello World!"}


app = FastAPI()
app.include_router(router)
