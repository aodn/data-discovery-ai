import unittest

from fastapi import FastAPI

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


if __name__ == "__main__":
    unittest.main()
