import asyncio
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI

from data_discovery_ai import server


class TestServerLifespan(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.state.model_status = "STARTING"
        self.app.state.model_error = None

    @patch("data_discovery_ai.server.load_nli_tokenizer_model")
    @patch("data_discovery_ai.server.load_embedding_tokenizer_model")
    async def test_load_models_background_success(self, mock_embedding, mock_nli):
        mock_embedding.return_value = ("tokenizer", "embedding_model")
        mock_nli.return_value = ("nli_tokenizer", "nli_model")

        await server.load_models_background(self.app)

        self.assertEqual(self.app.state.model_status, "UP")
        self.assertEqual(self.app.state.tokenizer, "tokenizer")
        self.assertEqual(self.app.state.embedding_model, "embedding_model")
        self.assertEqual(self.app.state.nli_tokenizer, "nli_tokenizer")
        self.assertEqual(self.app.state.nli_model, "nli_model")

    @patch("data_discovery_ai.server.load_nli_tokenizer_model")
    @patch("data_discovery_ai.server.load_embedding_tokenizer_model")
    async def test_load_models_background_failure_does_not_raise(
        self, mock_embedding, mock_nli
    ):
        mock_embedding.side_effect = OSError("connection reset")

        await server.load_models_background(self.app)

        self.assertEqual(self.app.state.model_status, "DOWN")
        self.assertIn("connection reset", self.app.state.model_error)
        mock_nli.assert_not_called()

    @patch("data_discovery_ai.server.load_llm_client", return_value=MagicMock())
    @patch(
        "data_discovery_ai.server.create_es_index", return_value=(MagicMock(), "idx")
    )
    @patch(
        "data_discovery_ai.server.load_nli_tokenizer_model",
        return_value=("nli_tokenizer", "nli_model"),
    )
    @patch("data_discovery_ai.server.load_embedding_tokenizer_model")
    async def test_lifespan_does_not_wait_for_models(
        self, mock_embedding, mock_nli, mock_es, mock_llm
    ):
        release = asyncio.Event()
        loop = asyncio.get_running_loop()

        def slow_load():
            # block the loader thread until the test releases it
            asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
            return "tokenizer", "embedding_model"

        mock_embedding.side_effect = slow_load

        async with server.lifespan(self.app):
            # startup completed while the models are still loading
            self.assertEqual(self.app.state.model_status, "STARTING")
            self.assertIsNone(self.app.state.tokenizer)
            release.set()

    @patch(
        "data_discovery_ai.server.create_es_index",
        side_effect=ConnectionError("es unreachable"),
    )
    async def test_lifespan_startup_failure_raises(self, mock_es):
        with self.assertRaises(ConnectionError):
            async with server.lifespan(self.app):
                pass


if __name__ == "__main__":
    unittest.main()
