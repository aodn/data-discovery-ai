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
                    "href": "https://example.com/notebook.ipynb",
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
                {
                    "href": "https://data-uplift-public.s3.ap-southeast-2.amazonaws.com/stored/datauplift/nrmn/nrmn.parquet",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"Data files via Amazon Web Services S3 storage - download link (full dataset)","description": "This is a data access link"}',
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
                {
                    "href": "http://acoustic.aodn.org.au",
                    "rel": "data",
                    "type": "",
                    "title": '"Acoustic Data Viewer (preview and download raw data)", "description": ""}',
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

    def test_python_notebook_recognised_by_title(self):
        """AC: R notebooks / video tutorials are discoverable under 'Code Tutorials'
        even when recognised by title alone (href has no .Rmd / is not youtube)."""
        request = {
            "links": [
                # R notebook: only the title signals it (AC-guaranteed wording)
                {
                    "href": "https://github.com/aodn/imos-user-code-library/blob/master/NESP/seabird",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Access to R Markdown notebook to query Cloud Optimised converted dataset",
                },
                # Video tutorial: only the title signals it (non-youtube href)
                {
                    "href": "https://example.com/tutorials/r-demo",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Video tutorials demonstrating the use of the associated R notebook",
                },
            ]
        }
        self.agent.execute(request)
        grouped = self.agent.response["links"]
        self.assertEqual(grouped[0]["ai:group"], "Code Tutorials")
        self.assertEqual(grouped[1]["ai:group"], "Code Tutorials")

    def test_python_notebook_real_links(self):
        """Real-world links from the AC: Jupyter (.ipynb), R Markdown (.Rmd) and the two
        YouTube video tutorials all group as 'Code Tutorials'. Only the .ipynb link gets
        the notebook MIME type; the R Markdown and video links keep their original type.
        """
        request = {
            "links": [
                {
                    "href": "https://github.com/aodn/imos-user-code-library/blob/master/NESP/seabird.ipynb",
                    "rel": "related",
                    "type": "application/x-ipynb+json",
                    "title": "Access to Jupyter notebook to query Cloud Optimised converted dataset",
                },
                {
                    "href": "https://github.com/aodn/imos-user-code-library/blob/master/NESP/seabird.Rmd",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Access to R Markdown notebook to query Cloud Optimised converted dataset",
                },
                # video tutorial whose title mentions "Jupyter notebook" but is a YouTube
                # link: must NOT be restamped with the notebook MIME type
                {
                    "href": "https://youtube.com/playlist?list=PLHCEbETnUz5w2KXiON-8iSYpO3x1qJgxj",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Video tutorials demonstrating the use of the associated Python Jupyter notebook",
                },
                {
                    "href": "https://youtube.com/playlist?list=PLHCEbETnUz5w0JYQ3BsDaMam-UZ42AA7w",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Video tutorials demonstrating the use of the associated R notebook",
                },
            ]
        }
        self.agent.execute(request)
        grouped = self.agent.response["links"]
        self.assertTrue(all(l["ai:group"] == "Code Tutorials" for l in grouped))
        self.assertEqual(grouped[0]["type"], "application/x-ipynb+json")
        self.assertEqual(grouped[1]["type"], "text/html")
        self.assertEqual(grouped[2]["type"], "text/html")
        self.assertEqual(grouped[3]["type"], "text/html")

    def test_placeholder_title_not_grouped(self):
        """Links whose title contains "(link to be added)" are placeholders and must not be grouped (hidden from portal)."""
        request = {
            "links": [
                {
                    "href": "https://github.com/aodn/imos-user-code-library",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title": "Access to Jupyter notebook to query Cloud Optimised converted dataset (link to be added)", "description": ""}',
                },
                # plain (non-JSON) title with the placeholder
                {
                    "href": "https://github.com/aodn/imos-user-code-library",
                    "rel": "related",
                    "type": "text/html",
                    "title": "Access to R Markdown notebook (link to be added)",
                },
                # real notebook link must still be grouped
                {
                    "href": "https://github.com/aodn/imos-user-code-library/blob/master/NESP/seabird.ipynb",
                    "rel": "related",
                    "type": "application/x-ipynb+json",
                    "title": "Access to Jupyter notebook to query Cloud Optimised converted dataset",
                },
            ]
        }
        self.agent.execute(request)
        links = self.agent.response["links"]
        self.assertNotIn("ai:group", links[0])
        self.assertNotIn("ai:group", links[1])
        self.assertEqual(links[2]["ai:group"], "Code Tutorials")

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
        self.assertEqual(len(result), 4)
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
        self.assertEqual(len(self.agent.response["links"]), 5)
        # expect output:
        # [{'href': 'https://example.com', 'rel': 'excluded_irrelated_link', 'type': 'text/html'}, {'href': 'https://example.ipynb', 'rel': 'related', 'application/x-ipynb+json', 'title': 'Example Notebook Link', 'group': 'Code Tutorials'}, {'href': 'https://example.com', 'rel': 'related', 'type': 'text/html', 'title': 'Example Document Link', 'group': 'Document'}, {'href': 'https://example.wms', 'rel': 'related', 'type': 'text/html', 'title': 'Example Data Link', 'group': 'Data Access'}]
        self.assertEqual(self.agent.response["links"][1]["ai:group"], "Code Tutorials")
        self.assertEqual(
            self.agent.response["links"][1]["type"], "application/x-ipynb+json"
        )

        self.assertEqual(
            self.agent.response["links"][4],
            {
                "href": "https://data-uplift-public.s3.ap-southeast-2.amazonaws.com/stored/datauplift/nrmn/nrmn.parquet",
                "rel": "related",
                "type": "text/html",
                "title": '{"Data files via Amazon Web Services S3 storage - download link (full dataset)","description": "This is a data access link"}',
                "ai:group": "Data Access",
                "ai:role": ["download"],
            },
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
                    "ai:role": ["download"],
                },
                {
                    "href": "https://data.imas.utas.edu.au/attachments/Abalone_habitat_warming_reefs/bathy/BLOCK27_bathy_50cm.tif",
                    "rel": "data",
                    "type": "",
                    "title": '{"title": "Block 27 - 50cm bathymetry [Geotiff DOWNLOAD], "description": "attachments"}',
                    "ai:group": "Data Access",
                    "ai:role": ["download"],
                },
                {
                    "href": "http://acoustic.aodn.org.au",
                    "rel": "data",
                    "type": "",
                    "title": '"Acoustic Data Viewer (preview and download raw data)", "description": ""}',
                    "ai:group": "Data Access",
                    "ai:role": ["download"],
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

    def test_execute_without_ai_role(self):
        # when no Data Access links exist, no ai:role should be tagged for links
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

        # sub_agent should have been initialised without calling execution
        self.assertIsNotNone(self.agent.sub_agent)

        # links must still exist in response with ai group mapping if works
        self.assertIn("links", self.agent.response)
        links = self.agent.response["links"]

        # verify grouping still works correctly
        pdf_link = next(l for l in links if "REPORT" in l.get("title", ""))
        self.assertEqual(pdf_link["ai:group"], "Document")

        other_link = next(l for l in links if "auswaves" in l.get("href", ""))
        self.assertEqual(other_link["ai:group"], "Other")

        # no link should have ai:role — sub_agent only runs on Data Access links
        for link in links:
            self.assertNotIn("ai:role", link)

    def test_execute_with_ai_role(self):
        # verify sub_agent response is merged into LinkGroupingAgent response
        request = {
            "links": [
                # thredds should be treated as downloadable link
                {
                    "href": "https://thredds.aodn.org.au/thredds/catalog/IMOS/catalog.html",
                    "rel": "related",
                    "type": "text/html",
                    "title": '{"title":"NetCDF files via THREDDS catalog","description":""}',
                },
                # wfs should be treated as downloadable link
                {
                    "href": "https://geoserver.imas.utas.edu.au/geoserver/seamap/wfs?version=1.0.0&request=GetFeature&typeName=SeamapAus_VIC_statewide_habitats_2023&outputFormat=SHAPE-ZIP",
                    "rel": "wfs",
                    "type": "",
                    "title": '{"title":"SHAPE-ZIP","description":"DATA ACCESS - This OGC WFS service returns the data (Statewide marine habitats 2023) in Shapefile format"}',
                },
                # wms should NOT be treated as downloadable link
                {
                    "href": "https://geoserver.imas.utas.edu.au/geoserver/seamap/wms",
                    "rel": "wms",
                    "type": "",
                    "title": '{"title":"seamap:SeamapAus_VIC_statewide_habitats_2023","description":"MAP - Statewide marine habitats 2023"}',
                },
                # visualisation link should NOT be treated as downloadable link
                {
                    "href": "https://www.marine.csiro.au/data/underway/?survey=in2019_v06",
                    "rel": "data",
                    "type": "",
                    "title": """{"title":"Underway Visualisation Tool","description":"Link to visualisation tool for Near Real-Time Underway Data"}""",
                },
                # survey link should NOT be treated as downloadable link
                {
                    "href": "https://www.marine.csiro.au/data/trawler/survey_details.cfm?survey=IN2019_V06",
                    "rel": "data",
                    "type": "",
                    "title": """{"title":"Investigator Survey","description":"Link to RV Investigator Survey Information, including voyage plans and summaries"}""",
                },
            ]
        }
        self.agent.execute(request)

        # 'links' should in agent's response
        self.assertIn("links", self.agent.response)
        links = self.agent.response["links"]

        # find links by href to verify ai:role assignment
        thredds_link = next(l for l in links if "thredds" in l.get("href", ""))
        wfs_link = next(l for l in links if "wfs" in l.get("rel", ""))
        wms_link = next(l for l in links if "wms" in l.get("rel", ""))
        visualisation_link = next(
            l for l in links if "Visualisation" in l.get("title", "")
        )
        survey_link = next(l for l in links if "Survey" in l.get("title", ""))

        # verify downloadable links have ai:role set correctly
        self.assertEqual(thredds_link["ai:role"], ["download"])
        self.assertEqual(wfs_link["ai:role"], ["download"])

        # WMS should not be marked as downloadable
        self.assertNotIn("ai:role", wms_link)

        # visualisaton/survey link should not be marked as downlaodable
        self.assertNotIn("ai:role", visualisation_link)
        self.assertNotIn("ai:role", survey_link)

        # verify temp flag is not leaked into output
        for link in links:
            self.assertNotIn("ai:is_page_downloadable", link)

    # --- AODN backlog #9106: "data access" titles and file-type suffix matching ---

    def _group_one(self, link):
        """Run the full grouping agent on a single fresh link and return the result."""
        self.agent.execute({"links": [dict(link)]})
        return self.agent.response["links"][0]

    def _assert_other_without_role(self, link):
        result = self._group_one(link)
        self.assertEqual(result["ai:group"], "Other", link["href"])
        self.assertNotIn("ai:role", result, link["href"])

    def _assert_download(self, link):
        result = self._group_one(link)
        self.assertEqual(result["ai:group"], "Data Access", link["href"])
        self.assertEqual(result["ai:role"], ["download"], link["href"])

    def test_csiro_ncmi_is_other(self):
        """'ncmi' in the URL must no longer match NetCDF files."""
        self._assert_other_without_role(
            {
                "href": "https://www.csiro.au/en/about/people/business-units/NCMI",
                "rel": "related",
                "type": "text/html",
                "title": "CSIRO NCMI",
            }
        )

    def test_data_access_title_is_downloadable(self):
        """A DOI landing page titled 'data access' is Data Access with a download role,
        for plain, mixed-case and combined JSON titles."""
        titles = [
            "Data Access Portal (data access)",
            "Data Access Portal (DATA Access)",
            '{"title": "Data Access Portal (data access)", "description": ""}',
        ]
        for title in titles:
            with self.subTest(title=title):
                self._assert_download(
                    {
                        "href": "https://doi.org/10.25919/ev38-0p48",
                        "rel": "related",
                        "type": "text/html",
                        "title": title,
                    }
                )

    def test_file_suffix_on_url_path_is_downloadable(self):
        hrefs = [
            "https://example.org/files/sample.nc",
            "https://example.org/files/sample.NC?token=abc",
            "https://example.org/files/sample.nc#section",
            "https://example.org/files/sample.csv",
            "https://example.org/files/sample.tiff",
            "https://example.org/files/store.zarr/",
        ]
        for href in hrefs:
            with self.subTest(href=href):
                self._assert_download(
                    {"href": href, "rel": "related", "type": "", "title": "Sample file"}
                )

    def test_file_suffix_outside_url_path_is_other(self):
        """Extensions in the host, query string, middle of the path or inside words do not match."""
        hrefs = [
            "https://example.org/files/sample.nc.bak",
            "https://example.org/page?name=sample.nc",
            "https://nc.example.org/page",
            "https://example.org/page?file=report.pdf",
            # 'tif' inside 'quantification' used to match as a GeoTIFF
            "https://www.nespmarine.edu.au/project/project-c5-quantification-risk-shipping-large-marine-fauna-across-australia",
        ]
        for href in hrefs:
            with self.subTest(href=href):
                self._assert_other_without_role(
                    {"href": href, "rel": "related", "type": "", "title": "Sample file"}
                )

    def test_file_suffix_on_url_path_other_groups(self):
        cases = [
            ("https://example.org/reports/final.PDF", "Document"),
            ("https://example.org/reports/final.docx", "Document"),
            # some stored hrefs are wrapped in literal quotes
            (
                '"https://nbviewer.org/github/aodn/aodn_cloud_optimised/blob/main/notebooks/radar.ipynb"',
                "Code Tutorials",
            ),
        ]
        for href, group in cases:
            with self.subTest(href=href):
                result = self._group_one(
                    {"href": href, "rel": "related", "type": "", "title": "Sample file"}
                )
                self.assertEqual(result["ai:group"], group)

    def test_wfs_output_format_still_downloadable(self):
        """Formats in WFS query strings are no longer suffix matches, but the wfs/outputformat rules still apply."""
        result = self._group_one(
            {
                "href": "https://geoserver.imas.utas.edu.au/geoserver/redmap/wfs?version=1.0.0&request=GetFeature&typeName=RM_SIGHTING_VIEW_DATA&outputFormat=CSV",
                "rel": "related",
                "type": "",
                "title": "Redmap sightings",
            }
        )
        self.assertEqual(result["ai:group"], "Data Access > wfs")
        self.assertEqual(result["ai:role"], ["download"])

    def test_downloadable_agent_suffix_outside_url_path(self):
        """Exercise the download agent directly, since grouping as Other would skip it."""
        hrefs = [
            "https://example.org/files/sample.nc.bak",
            "https://example.org/page?name=sample.nc",
            "https://nc.example.org/page",
            "https://www.csiro.au/en/about/people/business-units/NCMI",
        ]
        for href in hrefs:
            with self.subTest(href=href):
                link = self.agent.sub_agent.execute(
                    {
                        "link": {
                            "href": href,
                            "rel": "related",
                            "title": "Sample file",
                            "ai:group": "Data Access",
                        }
                    }
                )
                self.assertNotIn("ai:role", link)

        link = self.agent.sub_agent.execute(
            {
                "link": {
                    "href": "https://example.org/files/sample.NC?token=abc",
                    "rel": "related",
                    "title": "Sample file",
                    "ai:group": "Data Access",
                }
            }
        )
        self.assertEqual(link["ai:role"], ["download"])

    def test_data_access_title_respects_download_exclusions(self):
        links = [
            # wms subgroup
            {
                "href": "https://example.org/geoserver/wms",
                "title": "Data access",
                "ai:group": "Data Access > wms",
            },
            # visualisation / visualization titles
            {
                "href": "https://example.org/tool",
                "title": "Data access visualisation tool",
                "ai:group": "Data Access",
            },
            {
                "href": "https://example.org/tool",
                "title": "Data access visualization tool",
                "ai:group": "Data Access",
            },
            # survey description
            {
                "href": "https://example.org/voyage",
                "title": "Data access",
                "description": "Voyage survey information",
                "ai:group": "Data Access",
            },
            # invalid URLs
            {"href": "not a url", "title": "Data access", "ai:group": "Data Access"},
            {
                "href": "https://example.org\\data\\file.nc",
                "title": "Data access",
                "ai:group": "Data Access",
            },
        ]
        for link in links:
            with self.subTest(link=link):
                result = self.agent.sub_agent.execute({"link": dict(link)})
                self.assertNotIn("ai:role", result)
