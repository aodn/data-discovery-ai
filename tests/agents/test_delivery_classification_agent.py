# The unit test for the delivery mode classification agent in model/deliveryClassificationAgent.py
import unittest
from unittest.mock import patch, MagicMock
from data_discovery_ai.agents.deliveryClassificationAgent import (
    DeliveryClassificationAgent,
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

    @patch(
        "data_discovery_ai.agents.deliveryClassificationAgent.mapping_update_frequency"
    )
    @patch("data_discovery_ai.agents.deliveryClassificationAgent.logger")
    def test_execute_with_NIL(self, mock_logger, mock_mapping):
        # Mock rule-based path to return OTHER, so NLI path is triggered
        mock_mapping.return_value = UpdateFrequency.OTHER.value
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
        mock_mapping.assert_called_once_with(
            request["status"], request["temporal"], request["title"]
        )
        self.agent.take_action.assert_called_once_with(
            request["title"], request["abstract"], request["lineage"]
        )
        # Response is now a plain mode string, not a dict with "mode" key
        self.assertEqual(
            self.agent.response, {"summaries.ai:update_frequency": "real-time"}
        )
        mock_logger.debug.assert_called_once()
        log_msg = mock_logger.debug.call_args[0][0]
        self.assertIn("delivery_classification agent finished", log_msg)
        self.assertIn("real-time", log_msg)

    @patch(
        "data_discovery_ai.agents.deliveryClassificationAgent.mapping_update_frequency"
    )
    @patch("data_discovery_ai.agents.deliveryClassificationAgent.logger")
    def test_execute_rule_based(self, mock_logger, mock_mapping):
        # Mock rule-based path to return COMPLETED, NLI path should NOT be triggered
        mock_mapping.return_value = UpdateFrequency.COMPLETED.value
        self.agent.make_decision = MagicMock(return_value=True)
        self.agent.take_action = MagicMock()
        request = {
            "title": "Test Title",
            "abstract": "Test abstract.",
            "lineage": "Test lineage.",
            "status": "completed",
            "temporal": [],
        }
        self.agent.execute(request)
        # take_action should NOT be called when rule-based path returns a result
        self.agent.take_action.assert_not_called()
        self.assertEqual(
            self.agent.response,
            {"summaries.ai:update_frequency": UpdateFrequency.COMPLETED.value},
        )
