#!/bin/bash
echo "Loading environment variables from .env..."
if [ ! -f .env ]; then
  echo "Error: .env file not found. Please copy it from .env.sample"
  exit 1
fi

set -a
# shellcheck source=.env
source .env
set +a

echo "Exporting API Key..."
echo "OPENAI_API_KEY=$OPENAI_API_KEY"

echo "Starting MLFLOW Gateway..."
nohup mlflow gateway start --config-path mlflow_config.yaml --port 53001 > gateway.log 2>&1 &
sleep 5

export MLFLOW_DEPLOYMENTS_TARGET="http://127.0.0.1:53001"

echo "Starting MLFLOW Server..."
mlflow server --port 53000
