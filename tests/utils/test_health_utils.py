import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI

from data_discovery_ai.utils import health_utils
from data_discovery_ai.utils.health_utils import aggregate_status, check_elasticsearch


class TestHealthUtils(unittest.TestCase):
    def test_check_elasticsearch_defaults_to_down(self):
        self.assertEqual(
            check_elasticsearch(FastAPI()), {"status": "DOWN", "detail": None}
        )

    def test_check_elasticsearch_reflects_app_state(self):
        app = FastAPI()
        app.state.es_status = "STARTING"
        app.state.es_error = "retrying"

        self.assertEqual(
            check_elasticsearch(app),
            {"status": "STARTING", "detail": "retrying"},
        )

    def test_aggregate_status(self):
        up = {"status": "UP", "detail": None}
        starting = {"status": "STARTING", "detail": None}
        down = {"status": "DOWN", "detail": "error"}
        self.assertEqual(aggregate_status({"a": up, "b": up}), "UP")
        self.assertEqual(aggregate_status({"a": up, "b": starting}), "STARTING")
        self.assertEqual(aggregate_status({"a": starting, "b": down}), "DOWN")


class TestHealthPayload(unittest.IsolatedAsyncioTestCase):
    async def test_build_health_payload_for_each_aggregate_status(self):
        cases = (
            (
                {
                    "models": {"status": "UP", "detail": None},
                    "elasticsearch": {"status": "UP", "detail": None},
                },
                "UP",
            ),
            (
                {
                    "models": {"status": "STARTING", "detail": None},
                    "elasticsearch": {"status": "UP", "detail": None},
                },
                "STARTING",
            ),
            (
                {
                    "models": {"status": "DOWN", "detail": "failed"},
                    "elasticsearch": {"status": "UP", "detail": None},
                },
                "DOWN",
            ),
        )

        for components, expected_status in cases:
            with self.subTest(status=expected_status), patch.object(
                health_utils,
                "collect_components",
                new=AsyncMock(return_value=components),
            ):
                payload = await health_utils.build_health_payload(FastAPI())

            self.assertEqual(
                payload,
                {
                    "status_code": 200,
                    "status": expected_status,
                    "components": components,
                },
            )

    async def test_build_health_payload_reports_collection_error(self):
        with patch.object(
            health_utils,
            "collect_components",
            new=AsyncMock(side_effect=RuntimeError("check failed")),
        ):
            payload = await health_utils.build_health_payload(FastAPI())

        self.assertEqual(payload["status"], "DOWN")
        self.assertEqual(
            payload["components"]["health_check"],
            {"status": "DOWN", "detail": "check failed"},
        )

    async def test_build_health_payload_without_live_checks_skips_llm(self):
        app = FastAPI()
        app.state.model_status = "UP"
        app.state.es_status = "UP"
        mock_llm = AsyncMock()
        with patch.object(
            health_utils,
            "check_keyword_resources",
            return_value={"status": "UP", "detail": None},
        ), patch.object(health_utils, "check_llm", new=mock_llm):
            payload = await health_utils.build_health_payload(app, include_live=False)

        mock_llm.assert_not_awaited()
        self.assertEqual(payload["status"], "UP")
        self.assertEqual(
            list(payload["components"]),
            ["keyword_resources", "models", "elasticsearch"],
        )

    async def test_write_health_file_excludes_live_checks(self):
        payload = {"status_code": 200, "status": "STARTING", "components": {}}
        mock_build = AsyncMock(return_value=payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            health_file = os.path.join(temp_dir, "health.json")
            with patch.object(health_utils, "HEALTH_FILE", health_file), patch.object(
                health_utils, "build_health_payload", new=mock_build
            ):
                app = FastAPI()
                await health_utils.write_health_file(app)

        mock_build.assert_awaited_once_with(app, include_live=False)

    async def test_write_health_file_creates_directory_and_writes_payload(self):
        payload = {
            "status_code": 200,
            "status": "STARTING",
            "components": {"models": {"status": "STARTING", "detail": None}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            health_file = os.path.join(temp_dir, "status", "health.json")
            with patch.object(health_utils, "HEALTH_FILE", health_file), patch.object(
                health_utils,
                "build_health_payload",
                new=AsyncMock(return_value=payload),
            ):
                await health_utils.write_health_file(FastAPI())

            with open(health_file, encoding="utf-8") as file:
                self.assertEqual(json.load(file), payload)
            self.assertFalse(os.path.exists(f"{health_file}.tmp"))

    async def test_write_health_file_swallows_write_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            health_file = os.path.join(temp_dir, "status", "health.json")
            with patch.object(health_utils, "HEALTH_FILE", health_file), patch.object(
                health_utils.os,
                "makedirs",
                side_effect=OSError("read-only filesystem"),
            ), patch.object(health_utils.logger, "error") as mock_log:
                await health_utils.write_health_file(FastAPI())

            mock_log.assert_called_once()

    def test_remove_health_file_deletes_file_and_tolerates_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            health_file = os.path.join(temp_dir, "health.json")
            with open(health_file, "w", encoding="utf-8") as file:
                file.write("{}")

            with patch.object(health_utils, "HEALTH_FILE", health_file):
                health_utils.remove_health_file()
                self.assertFalse(os.path.exists(health_file))
                health_utils.remove_health_file()


if __name__ == "__main__":
    unittest.main()
