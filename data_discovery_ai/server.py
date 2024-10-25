from fastapi import FastAPI
from data_discovery_ai.routes import router as api_router


app = FastAPI()
app.include_router(api_router)
