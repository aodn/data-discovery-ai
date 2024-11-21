API_PREFIX = "/api/v1/ml"
API_KEY_NAME = "X-API-Key"
AVAILABLE_MODELS = ["development", "staging", "production", "experimental", "benchmark"]
KEYWORD_CONFIG = "keyword_classification_parameters.ini"
ELASTICSEARCH_CONFIG = "esManager.ini"
KEYWORD_SAMPLE_FILE = "keyword_sample.pkl"
KEYWORD_LABEL_FILE = "keyword_label.pkl"

#  global constants for es_connector
BATCH_SIZE = 100
SLEEP_TIME = 5
ES_INDEX_NAME = "es-indexer-staging"

# global constants for preprocessor
RARE_LABEL_THRESHOLD = 5
