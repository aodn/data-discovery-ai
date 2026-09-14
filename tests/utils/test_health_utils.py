import unittest

from data_discovery_ai.utils.health_utils import aggregate_status


class TestHealthUtils(unittest.TestCase):
    def test_aggregate_status(self):
        up = {"status": "UP", "detail": None}
        starting = {"status": "STARTING", "detail": None}
        down = {"status": "DOWN", "detail": "error"}
        self.assertEqual(aggregate_status({"a": up, "b": up}), "UP")
        self.assertEqual(aggregate_status({"a": up, "b": starting}), "STARTING")
        self.assertEqual(aggregate_status({"a": starting, "b": down}), "DOWN")


if __name__ == "__main__":
    unittest.main()
