#!/bin/sh

poetry run uvicorn --host 0.0.0.0 data_discovery_ai.server:app --log-config=log_config.yaml
