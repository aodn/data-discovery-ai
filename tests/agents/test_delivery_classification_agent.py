# The unit test for the delivery mode classification agent in model/deliveryClassificationAgent.py
import unittest
from unittest.mock import patch, MagicMock
from data_discovery_ai.agents.deliveryClassificationAgent import (
    DeliveryClassificationAgent,
    map_status_update_frequency,
    UpdateFrequency,
)


class TestDeliveryClassificationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DeliveryClassificationAgent()
        self.agent.set_required_fields(
            ["title", "abstract", "lineage", "status", "temporal"]
        )
        self.valid_request = {
            "title": "Test Title",
            "abstract": "Test abstract.",
            "lineage": "Test lineage.",
            "status": "on going",
            "temporal": [],
        }

        self.invalid_request = {"title": "Test Title", "abstract": "Test abstract."}

    def test_make_decision(self):
        # Test valid request
        self.assertTrue(self.agent.make_decision(self.valid_request))

        # Test invalid request with missing field
        self.assertFalse(self.agent.make_decision(self.invalid_request))

    @patch("data_discovery_ai.agents.deliveryClassificationAgent.logger")
    def test_execute(self, mock_logger):
        self.agent.make_decision = MagicMock(return_value=True)
        self.agent.take_action = MagicMock(return_value="real-time")

        request = {
            "title": "Test Title",
            "abstract": "Test abstract.",
            "lineage": "Test lineage.",
            "status": "on going",
            "temporal": [],
        }

        self.agent.execute(request)

        self.agent.make_decision.assert_called_once_with(request)
        self.agent.take_action.assert_called_once_with(
            request["title"], request["abstract"], request["lineage"]
        )
        self.assertEqual(
            self.agent.response, {"summaries.ai:update_frequency": "real-time"}
        )

        # Check logging
        mock_logger.debug.assert_called_once()
        log_msg = mock_logger.debug.call_args[0][0]
        self.assertIn("delivery_classification agent finished", log_msg)
        self.assertIn("real-time", log_msg)

    def test_map_status_update_frequency(self):
        completed_status = "historicalArchive"
        completed_temporal = [
            {"start": "2023-01-22T13:00:00Z", "end": "2023-01-23T12:59:59Z"}
        ]
        ongoing_temporal = [{"start": "2023-01-22T13:00:00Z"}]
        self.assertEqual(
            map_status_update_frequency(completed_status, completed_temporal),
            UpdateFrequency.completed.value,
        )

        free_text_status = "Under development"
        self.assertEqual(
            map_status_update_frequency(free_text_status, completed_temporal),
            UpdateFrequency.completed.value,
        )
        self.assertEqual(
            map_status_update_frequency(free_text_status, ongoing_temporal),
            UpdateFrequency.other.value,
        )

        ongoing_status = "onGoing | historicalArchive"
        self.assertEqual(
            map_status_update_frequency(ongoing_status, ongoing_temporal),
            UpdateFrequency.other.value,
        )
