import json
import os
import tempfile
import unittest
from unittest.mock import patch

from data_discovery_ai.utils.health_utils import (
    aggregate_status,
    remove_health_status,
    write_health_status,
)


class TestHealthUtils(unittest.TestCase):
    def test_aggregate_status(self):
        up = {"status": "UP", "detail": None}
        starting = {"status": "STARTING", "detail": None}
        down = {"status": "DOWN", "detail": "error"}
        self.assertEqual(aggregate_status({"a": up, "b": up}), "UP")
        self.assertEqual(aggregate_status({"a": up, "b": starting}), "STARTING")
        self.assertEqual(aggregate_status({"a": starting, "b": down}), "DOWN")

    def test_write_and_remove_health_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            health_json = os.path.join(tmp_dir, "status", "health.json")
            with patch("data_discovery_ai.utils.health_utils.HEALTH_JSON", health_json):
                write_health_status("STARTING")
                with open(health_json) as f:
                    self.assertEqual(
                        json.load(f), {"status": "STARTING", "status_code": 200}
                    )
                self.assertFalse(os.path.exists(f"{health_json}.tmp"))

                components = {"models": {"status": "UP", "detail": None}}
                write_health_status("UP", components)
                with open(health_json) as f:
                    self.assertEqual(json.load(f)["components"], components)

                remove_health_status()
                self.assertFalse(os.path.exists(health_json))
                # removing a missing file does not raise
                remove_health_status()

    def test_write_health_status_never_raises(self):
        with patch(
            "data_discovery_ai.utils.health_utils.HEALTH_JSON",
            "/proc/not-writable/health.json",
        ):
            write_health_status("UP")


if __name__ == "__main__":
    unittest.main()
