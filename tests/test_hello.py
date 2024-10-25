import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from data_discovery_ai.common.constants import API_PREFIX
from data_discovery_ai.server import app
from data_discovery_ai.utils.api_utils import API_KEY

client = TestClient(app)


@pytest.mark.asyncio
async def test_hello_authenticated(monkeypatch):
    headers = {"X-API-KEY": API_KEY}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        response = await async_client.get(f"{API_PREFIX}/hello", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"content": "Hello World!"}


@pytest.mark.asyncio
async def test_hello_invalid_api_key():
    headers = {"X-API-Key": "invalid_key"}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        response = await async_client.get(f"{API_PREFIX}/hello", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"}


@pytest.mark.asyncio
async def test_hello_missing_api_key():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        response = await async_client.get(f"{API_PREFIX}/hello")

    assert response.status_code == 403
    assert response.json() == {"detail": "Not authenticated"}
