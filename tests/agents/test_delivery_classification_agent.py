# The unit test for the delivery mode classification agent in model/deliveryClassificationAgent.py
import unittest
from unittest.mock import patch, MagicMock
from data_discovery_ai.agents.deliveryClassificationAgent import (
    DeliveryClassificationAgent,
)


class TestDeliveryClassificationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DeliveryClassificationAgent()
        self.agent.set_required_fields(["title", "abstract", "lineage"])
        self.valid_request = {
            "title": "Test Title",
            "abstract": "Test abstract.",
            "lineage": "Test lineage.",
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
            "title": "Test title",
            "abstract": "Test abstract",
            "lineage": "test lineage",
        }

        self.agent.execute(request)

        self.agent.make_decision.assert_called_once_with(request)
        self.agent.take_action.assert_called_once_with(
            request["title"], request["abstract"], request["lineage"]
        )
        self.assertEqual(self.agent.response, {"ai_update_frequency": "real-time"})

        # Check logging
        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]
        self.assertIn("delivery_classification agent finished", log_msg)
        self.assertIn("real-time", log_msg)
