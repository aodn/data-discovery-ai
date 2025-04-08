API_PREFIX = "/api/v1/ml"
API_KEY_NAME = "X-API-Key"
# remove the following line as we get the value from the environment.
AVAILABLE_MODELS = ["development", "staging", "production", "experimental", "benchmark"]
API_VALID_FIELD = ["title", "abstract", "lineage", "links"]
AVAILABLE_AI_MODELS = ["keyword classification", "data delivery mode classification", "description formating", "link grouping"]

MODEL_CONFIG = "classification_parameters.ini"
ELASTICSEARCH_CONFIG = "esManager.ini"
KEYWORD_FOLDER = "KeywordClassifier"
KEYWORD_SAMPLE_FILE = "keyword_sample.pkl"
KEYWORD_LABEL_FILE = "keyword_label.pkl"
FILTER_PREPROCESSED_FILE = "filter_preprocessed.pkl"
FILTER_FOLDER = "DataDeliveryModeFilter"

# TODO: move to es config file
#  global constants for es_connector


# global constants for preprocessor
RARE_LABEL_THRESHOLD = 3

# TODO: move to model parameter config file
# predefined decision-making rules for link grouping
GROUPING_RULES = {
    # Rule 1: if the href contains "ipynb", then it is a Python notebook
    "Python Notebook": {"href": ["ipynb"]},
    # Rule 2: if the href contains "pdf"/"doc", then it is a document
    "Document": {
        "href": ["pdf", "doc"],
        # Rule 3: if the title contains document keywords, then it is a document
        "title": [
            "document",
            "documentation",
            "manual",
            "support",
            "guide",
            "help",
            "report",
        ],
    },
    "Data Access": {
        # Rule 4: if the href contains data access keywords, then it is a data access link. Data access keywords includes:
        #   (1) text keywords: "access", "download"
        #   (2) format keywords: "csv", "json", "nc", "zip"
        #   (3) service keywords: "wms", "wfs", "ows"
        #   (4) source keywords: "thredds", "trawler"
        #   (5) parameter keywords: "data_type", "download", "format"
        "href": [
            "access",
            "download",
            "csv",
            "xlsx",
            "json",
            "nc",
            "zip",
            "wms",
            "wfs",
            "ows",
            "thredds",
            "trawler",
            "data_type",
            "format",
        ],
        # Rule 5: if the title contains data access keywords, then it is a data access link
        "title": ["data access", "download data", "access data", "data download"],
        # Rule 6: if the content contains data access keywords, then it is a data access link, use a combination of keywords as we don't know the exact content
        "content": [["data", "dataset"], ["download"]],
    },
}
