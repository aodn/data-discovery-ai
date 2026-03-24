# unit test for the link grouping agent in model/linkGroupingAgent.py
import unittest
from unittest.mock import patch, MagicMock
import requests

from data_discovery_ai.agents.linkGroupingAgent import (
    LinkGroupingAgent,
    subgroup_access_link,
)
from data_discovery_ai.config.config import ConfigUtil


class TestLinkGroupingAgent(unittest.TestCase):
    def setUp(self):
        self.agent = LinkGroupingAgent()
        self.agent.set_required_fields(["links"])
        self.valid_request = {
            "links": [
                {
                    "href": "https://example.com",
                    "rel": "excluded_irrelated_link",
                    "type": "text/html",
                },
                {
                    "href": "https://example.ipynb",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "Example Notebook Link", "description": "This is a Python Notebook link"}',
                },
                {
                    "href": "https://example.com",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "Example Document Link", "description": "This is a document link"}',
                },
                {
                    "href": "https://example.wms",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "Example Data Link", "description": "This is a data access link"}',
                },
            ]
        }

        self.valid_protocol_request = {
            "links": [
                {
                    "href": "http://nesptropical.edu.au/wp-content/uploads/2016/03/NESP-TWQ-3.1-FINAL-REPORT.pdf",
                    "rel": "WWW:LINK-1.0-http--publication",
                    "type": "",
                    "title": '{"title": "REPORT - Project Final Report[PDF]", "description": ""}',
                },
                {
                    "href": "https://catalogue.aodn.org.au:443/geonetwork/srv/api/records/05818c50-14c2-11dd-bdaa-00188b4c0af8/attachments/1989_01_12.zip",
                    "rel": "data",
                    "type": "",
                    "title": '{"title": "1989_01_12.zip", "description": ""}',
                },
                {
                    "href": "https://data.imas.utas.edu.au/attachments/Abalone_habitat_warming_reefs/bathy/BLOCK27_bathy_50cm.tif",
                    "rel": "data",
                    "type": "",
                    "title": '{"title": "Block 27 - 50cm bathymetry [Geotiff DOWNLOAD], "description": "attachments"}',
                },
            ]
        }

        self.hide_protocol_request = {
            "links": [
                {
                    "href": "https://processes.aodn.org.au/wps",
                    "rel": "OGC:WPS--gogoduck",
                    "type": "",
                    "title": '{"title": "csiro_oa_reconstruction_url", "description": "A wms layer name"}',
                },
                {
                    "href": "https://help.aodn.org.au/web-services/gogoduck-aggregator/",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "GoGoDuck help documentation", "description": ""}',
                },
                {
                    "href": "https://portal.aodn.org.au/search?uuid=7709f541-fc0c-4318-b5b9-9053aa474e0e",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "View and download data though the AODN Portal", "description": ""}',
                },
                {
                    "href": "https://help.aodn.org.au/web-services/ncurllist-service/",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "ncUrlList help documentation", "description": ""}',
                },
                {
                    "href": "http://geoserver-123.aodn.org.au/geoserver/ows",
                    "rel": "IMOS:AGGREGATION--bodaac",
                    "type": "",
                    "title": '{"title": "anmn_velocity_timeseries_map#file_url", "description": "A wms layer name"}',
                },
            ]
        }

        self.invalid_request = {
            "links": [
                {
                    "href": "https://example.com",
                    "rel": "parent",
                    "type": "text/html",
                    "title": "Example Link",
                }
            ]
        }

    def test_subgroup_access_link(self):
        data_access_links = [
            # subgroup: thredds
            {
                "href": "http://thredds.aodn.org.au/thredds/catalog/IMOS/Argo/dac/catalog.html",
                "rel": "related",
                "type": "text/html",
                "title": '{"title": "NetCDF files via THREDDS catalog", "description": ""}',
                "ai:group": "Data Access",
            },
            # subgroup: wms
            {
                "href": "http://geoserver-123.aodn.org.au/geoserver/wms",
                "rel": "wms",
                "type": "",
                "title": '{"title": "imos:argo_profile_bio_map", "description": "A wms layer name"}',
                "ai:group": "Data Access",
            },
            # subgroup: aws
            {
                "href": "https://registry.opendata.aws/aodn_slocum_glider_delayed_qc/",
                "rel": "related",
                "type": "text/html",
                "title": '{"title": "Access To AWS Open Data Program registry for the Cloud Optimised version of this dataset", "description": ""}',
                "ai:group": "Data Access",
            },
            # wfs
            {
                "href": "http://geoserver-123.aodn.org.au/geoserver/ows",
                "rel": "wfs",
                "type": "",
                "title": '{"title": "anfog_dm_trajectory_data", "description": "A wms layer name"}',
                "ai:group": "Data Access",
            },
            # no specific subgroup
            {
                "href": "https://www.marine.csiro.au/data/trawler/dataset.cfm?survey=IN2015_V02&data_type=adcp",
                "rel": "data",
                "type": "",
                "title": '{"title": "Data available via Data Trawler", "description": "A data trawler link"}',
                "ai:group": "Data Access",
            },
        ]

        result = [subgroup_access_link(link) for link in data_access_links]

        self.assertEqual(result[0]["ai:group"], "Data Access > thredds")  # thredds
        self.assertEqual(result[1]["ai:group"], "Data Access > wms")  # wms
        self.assertEqual(result[2]["ai:group"], "Data Access > aws")  # aws
        self.assertEqual(result[3]["ai:group"], "Data Access > wfs")  # wfs
        self.assertEqual(result[4]["ai:group"], "Data Access")  # no specific subgroup

    @patch("data_discovery_ai.agents.linkGroupingAgent.requests.get")
    def test_ungrouped_links_with_fallback(self, mock_get):
        valid_data_link = {
            "href": "https://example.tif",
            "rel": "data",
            "type": "",
            "title": '{"title": "Example Image Data Link", "description": ""}',
        }

        grouped_valid_data_link = self.agent.grouping(valid_data_link, [])
        self.assertEqual(grouped_valid_data_link, "Data Access")

        valid_crawl_link = {
            "href": "https://data.org/page.html",
            "rel": "related",
            "title": '{"title": "some title", "description": ""}',
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "This page has dataset access"
        mock_get.return_value = mock_resp

        keywords = ["dataset access"]
        grouped_valid_crawl_link = self.agent.grouping(valid_crawl_link, keywords)
        self.assertEqual(grouped_valid_crawl_link, "Data Access")

        invalid_crawl_link = {
            "href": "https://data.org/page.html",
            "rel": "related",
            "title": '{"title": "some title", "description": ""}',
        }
        mock_get.side_effect = requests.exceptions.Timeout

        keywords = ["dataset access"]
        grouped_invalid_crawl_link = self.agent.grouping(invalid_crawl_link, keywords)
        self.assertEqual(grouped_invalid_crawl_link, "Other")

    def test_make_decision(self):
        result = self.agent.make_decision(self.valid_request)
        # expect to skip the first irrelated link
        self.assertEqual(len(result), 3)
        self.assertEqual(
            result[0]["title"],
            '{"title": "Example Notebook Link", "description": "This is a Python Notebook link"}',
        )

    def test_make_decision_invalid_request(self):
        result = self.agent.make_decision(self.invalid_request)
        self.assertEqual(result, [])

    def test_load_config(self):
        # Test loading configuration
        self.assertIsInstance(self.agent.config, ConfigUtil)
        self.assertTrue(self.agent.model_config)
        self.assertIn("grouping_rules", self.agent.model_config)

    def test_execute(self):
        self.agent.execute(self.valid_request)
        # it should return all links with selected links to be grouped
        self.assertEqual(len(self.agent.response["links"]), 4)
        # expect output:
        # [{'href': 'https://example.com', 'rel': 'excluded_irrelated_link', 'type': 'text/html'}, {'href': 'https://example.ipynb', 'rel': 'related', 'application/x-ipynb+json', 'title': 'Example Notebook Link', 'group': 'Python Notebook'}, {'href': 'https://example.com', 'rel': 'related', 'type': 'text/html', 'title': 'Example Document Link', 'group': 'Document'}, {'href': 'https://example.wms', 'rel': 'related', 'type': 'text/html', 'title': 'Example Data Link', 'group': 'Data Access'}]
        self.assertEqual(self.agent.response["links"][1]["ai:group"], "Python Notebook")
        self.assertEqual(
            self.agent.response["links"][1]["type"], "application/x-ipynb+json"
        )

    def test_execute_with_protocol(self):
        self.agent.execute(self.valid_protocol_request)
        self.assertEqual(
            self.agent.response["links"],
            [
                {
                    "href": "http://nesptropical.edu.au/wp-content/uploads/2016/03/NESP-TWQ-3.1-FINAL-REPORT.pdf",
                    "rel": "WWW:LINK-1.0-http--publication",
                    "type": "",
                    "title": '{"title": "REPORT - Project Final Report[PDF]", "description": ""}',
                    "ai:group": "Document",
                },
                {
                    "href": "https://catalogue.aodn.org.au:443/geonetwork/srv/api/records/05818c50-14c2-11dd-bdaa-00188b4c0af8/attachments/1989_01_12.zip",
                    "rel": "data",
                    "type": "",
                    "title": '{"title": "1989_01_12.zip", "description": ""}',
                    "ai:group": "Data Access",
                },
                {
                    "href": "https://data.imas.utas.edu.au/attachments/Abalone_habitat_warming_reefs/bathy/BLOCK27_bathy_50cm.tif",
                    "rel": "data",
                    "type": "",
                    "title": '{"title": "Block 27 - 50cm bathymetry [Geotiff DOWNLOAD], "description": "attachments"}',
                    "ai:group": "Data Access",
                },
            ],
        )

    def test_execute_with_hidden_protocol(self):
        self.agent.execute(self.hide_protocol_request)
        # non of them should have `ai:group` field
        self.assertEqual(
            self.agent.response["links"],
            [
                {
                    "href": "https://processes.aodn.org.au/wps",
                    "rel": "OGC:WPS--gogoduck",
                    "type": "",
                    "title": '{"title": "csiro_oa_reconstruction_url", "description": "A wms layer name"}',
                },
                {
                    "href": "https://help.aodn.org.au/web-services/gogoduck-aggregator/",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "GoGoDuck help documentation", "description": ""}',
                },
                {
                    "href": "https://portal.aodn.org.au/search?uuid=7709f541-fc0c-4318-b5b9-9053aa474e0e",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "View and download data though the AODN Portal", "description": ""}',
                },
                {
                    "href": "https://help.aodn.org.au/web-services/ncurllist-service/",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "ncUrlList help documentation", "description": ""}',
                },
                {
                    "href": "http://geoserver-123.aodn.org.au/geoserver/ows",
                    "rel": "IMOS:AGGREGATION--bodaac",
                    "type": "",
                    "title": '{"title": "anmn_velocity_timeseries_map#file_url", "description": "A wms layer name"}',
                },
            ],
        )

    def test_execute_without_ai_assets(self):
        # when no Data Access links exist, summaries.ai:assets should not appear in response
        request = {
            "links": [
                # document link — not downloadable data link
                {
                    "href": "http://nesptropical.edu.au/wp-content/uploads/2016/03/NESP-TWQ-3.1-FINAL-REPORT.pdf",
                    "rel": "WWW:LINK-1.0-http--publication",
                    "type": "",
                    "title": "REPORT - Project Final Report",
                },
                # other link — not downloadable data link
                {
                    "href": "https://auswaves.org/",
                    "rel": "related",
                    "type": "text/html",
                    "title": "AusWaves website",
                },
            ]
        }
        self.agent.execute(request)

        # links must still exist in response with ai group mapping if works
        self.assertIn("links", self.agent.response)

        links = self.agent.response["links"]
        pdf_link = next(l for l in links if "REPORT" in l.get("title", ""))
        self.assertEqual(pdf_link["ai:group"], "Document")
        other_link = next(l for l in links if "auswaves" in l.get("href", ""))
        self.assertEqual(other_link["ai:group"], "Other")

        # summaries.ai:assets must NOT be present when there are no downloadable links
        self.assertNotIn("summaries.ai:assets", self.agent.response)

        # sub_agent should not have been initialised
        self.assertIsNone(self.agent.sub_agent)

    def test_execute_produces_ai_assets(self):
        # verify sub_agent response is merged into LinkGroupingAgent response
        request = {
            "links": [
                # thredds should be treated as downloadable link
                {
                    "href": "https://thredds.aodn.org.au/thredds/catalog/IMOS/catalog.html",
                    "rel": "related",
                    "type": "text/html",
                    "title": "NetCDF files via THREDDS catalog",
                },
                # wfs should be treated as downloadable link
                {
                    "href": "https://geoserver.imas.utas.edu.au/geoserver/seamap/wfs?version=1.0.0&request=GetFeature&typeName=SeamapAus_VIC_statewide_habitats_2023&outputFormat=SHAPE-ZIP",
                    "rel": "wfs",
                    "type": "",
                    "title": "SHAPE-ZIP",
                    "description": "DATA ACCESS - This OGC WFS service returns the data (Statewide marine habitats 2023) in Shapefile format",
                },
                # wms should NOT be treated as downloadable link
                {
                    "href": "https://geoserver.imas.utas.edu.au/geoserver/seamap/wms",
                    "rel": "wms",
                    "type": "",
                    "title": "seamap:SeamapAus_VIC_statewide_habitats_2023",
                    "description": "MAP - Statewide marine habitats 2023",
                },
            ]
        }
        self.agent.execute(request)

        # summaries.ai:assets must exist in response
        self.assertIn("summaries.ai:assets", self.agent.response)

        assets = self.agent.response["summaries.ai:assets"]
        self.assertIsInstance(assets, dict)

        thredds_href = "https://thredds.aodn.org.au/thredds/catalog/IMOS/catalog.html"
        wfs_href = "https://geoserver.imas.utas.edu.au/geoserver/seamap/wfs?version=1.0.0&request=GetFeature&typeName=SeamapAus_VIC_statewide_habitats_2023&outputFormat=SHAPE-ZIP"
        wms_href = "https://geoserver.imas.utas.edu.au/geoserver/seamap/wms"

        # verify rules work
        self.assertIn(thredds_href, assets)
        self.assertIn(wfs_href, assets)

        self.assertNotIn(wms_href, assets)

        # verify asset structure
        for href, asset in assets.items():
            self.assertEqual(href, asset["href"])
            self.assertEqual(asset["role"], "DOWNLOAD")
            self.assertIn("title", asset)
            self.assertIn("description", asset)
            self.assertIn("type", asset)

        # verify temp flag is not leaked into output links
        for link in self.agent.response["links"]:
            self.assertNotIn("ai:is_page_downloadable", link)
