import gzip
import json
import os
from io import BytesIO
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from data_discovery_ai.server import app
from dotenv import load_dotenv

load_dotenv()

client = TestClient(app)


def setup_app_state_mocks():
    app.state.tokenizer = MagicMock()
    app.state.embedding_model = MagicMock()


def test_process_record_with_compressed_request():
    setup_app_state_mocks()

    payload = {
        "selected_model": ["link_grouping"],
        "uuid": "test-uuid",
        "links": [
            {
                "href": "https://example.com",
                "title": "Example Link",
                "rel": "related",
                "type": "text/html",
            }
        ],
    }

    # Compress payload
    compressed_body = BytesIO()
    with gzip.GzipFile(fileobj=compressed_body, mode="wb") as f:
        f.write(json.dumps(payload).encode("utf-8"))

    API_KEY = os.getenv("API_KEY")

    # Send request
    response = client.post(
        "/api/v1/ml/process_record",
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "X-API-Key": API_KEY,
        },
        data=compressed_body.getvalue(),
    )

    assert response.status_code == 200
