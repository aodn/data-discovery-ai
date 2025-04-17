API_PREFIX = "/api/v1/ml"
API_KEY_NAME = "X-API-Key"
AVAILABLE_MODELS = ["development", "staging", "production", "experimental", "benchmark"]
AVAILABLE_AI_AGENT = [
    {"model": "keyword_classification", "required fields": ["title", "abstract"]},
    {
        "model": "delivery_classification",
        "required fields": ["title", "abstract", "lineage"],
    },
    {"model": "description_formatting", "required fields": ["title", "abstract"]},
    {"model": "link_grouping", "required fields": ["links"]},
]
MAX_PROCESS = 4

PARAMETER_FILE = "parameters.yaml"
KEYWORD_FOLDER = "KeywordClassifier"
KEYWORD_SAMPLE_FILE = "keyword_sample.pkl"
KEYWORD_LABEL_FILE = "keyword_label.pkl"
FILTER_PREPROCESSED_FILE = "filter_preprocessed.pkl"
FILTER_FOLDER = "DataDeliveryModeFilter"

# global constants for preprocessor
RARE_LABEL_THRESHOLD = 3
