import unittest

from data_discovery_ai.enum.delivery_mode_enum import UpdateFrequency
from data_discovery_ai.ml.filteringModel import mapping_update_frequency

import data_discovery_ai.ml.filteringModel as filteringModel


class TestFilteringModel(unittest.TestCase):
    def test_map_title_update_frequency(self):
        real_time_in_title = "Wave buoys Observations - Australia - near real-time"
        self.assertEqual(
            filteringModel.map_title_update_frequency(real_time_in_title),
            UpdateFrequency.REAL_TIME.value,
        )

        delayed_in_title = "IMOS - Animal Tracking Facility - Satellite Relay Tagging Program - Delayed mode data"
        self.assertEqual(
            filteringModel.map_title_update_frequency(delayed_in_title),
            UpdateFrequency.DELAYED.value,
        )

        unknow_title = "Victorian Statewide Marine Habitat Map 2023"
        self.assertEqual(filteringModel.map_title_update_frequency(unknow_title), None)

    def test_map_status_update_frequency(self):
        completed_status = "historicalArchive"
        completed_temporal = [
            {"start": "2023-01-22T13:00:00Z", "end": "2023-01-23T12:59:59Z"}
        ]
        title = "IMOS - Animal Tracking Facility - Satellite Relay Tagging Program - Delayed mode data"
        ongoing_temporal = [{"start": "2023-01-22T13:00:00Z"}]
        self.assertEqual(
            mapping_update_frequency(completed_status, completed_temporal, title),
            UpdateFrequency.COMPLETED.value,
        )

        free_text_status = "Under development"
        self.assertEqual(
            mapping_update_frequency(free_text_status, completed_temporal, title),
            UpdateFrequency.COMPLETED.value,
        )
        self.assertEqual(
            mapping_update_frequency(free_text_status, ongoing_temporal, title),
            UpdateFrequency.DELAYED.value,
        )

        ongoing_status = "onGoing | historicalArchive"
        self.assertEqual(
            mapping_update_frequency(ongoing_status, ongoing_temporal, title),
            UpdateFrequency.DELAYED.value,
        )


if __name__ == "__main__":
    unittest.main()
