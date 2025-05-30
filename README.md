[![Language: Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![CI](https://github.com/aodn/data-discovery-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/aodn/data-discovery-ai/actions/workflows/ci.yml)

# Data Discovery AI
A consolidated framework for AI and ML components developed for the new portal.

## 🔧 Setup
### Environment variables
To run the app locally:
1. Go to the root folder of this project.
2. Create a .env file. (or simply copy the .env.sample file)
3. Add the following (with your real keys):
```shell
API_KEY="your_actual_api_key_here"
OPENAI_API_KEY="your_actual_openai_api_key_here"
PROFILE="your_actual_environment_here"
```
If you plan to train models, also include:
```shell
ES_ENDPOINT="your_actual_elasticsearch_endpoint"
ES_API_KEY="your_actual_es_api_key"
```

## 🚀 Running the App
### Option 1: Run with Docker (Recommended for Non-Developers)
1. Copy the content in .env.sample` to `.env` and fill in your keys.
2. Run
    ```shell
    ./startServer.sh
    ```
    Then visit [http://localhost:8000](http://localhost:8000)
3. Test App health at [http://localhost:8000/api/v1/ml/health](http://localhost:8000/api/v1/ml/health)

### Option 2: Run for Development (For Developers)
#### Conda (recommended for creating a virtual environment)
1. Install Conda (if not already installed):

   Follow the instructions at [Conda Installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Create Conda virtual environment:

   ```shell
   conda env create -f environment.yml
   ```

#### Dependencies management

Poetry is used for dependency management, the `pyproject.toml` file is what is the most important, it will orchestrate the project and its dependencies.

You can update the file `pyproject.toml` for adding/removing dependencies by using

```shell
poetry add <pypi-dependency-name> # e.g poetry add numpy
poetry remove <pypi-dependency-name> # e.g. poetry remove numpy
```

You might want to update the `poetry.lock` file after manually modifying `pyproject.toml` with `poetry lock` command. To update all dependencies, use `poetry update` command.

#### Installation and Usage

1. Activate Conda virtual environment:

   ```shell
   conda activate data-discovery-ai
   ```

2. Install environment dependencies:

   ```shell
   # after cloning the repo with git clone command
   cd data-discovery-ai
   poetry install
   ```

#### Before Usage
FastAPI runs internal checks before making `\process_record` API calls. These checks include:
1. ✅ Required model resource files must be present in `data_discovery_ai/resources/`.
2. ✅ A valid `OPENAI_API_KEY` must be in `.env` unless you're in `development` environment.
3. ✅ If `PROFILE=development`, Ollama must be running locally at http://localhost:11434.

##### (Optional) Install Llama3 for Local Development
To use the Llama3 model locally without OpenAI:
1. Go to [Ollama download page](https://ollama.com/download) and download the version that matches your operating system (Windows, Linux, or macOS).
2. After installation, start Ollama either by launching the app or running the following command:
    ```shell
    ollama serve
    ```
3. Pull the "llama3" model used for local development：
    ```shell
    ollama pull llama3
    ```

4. (Optional) Consider installing Open WebUI to test Llama3 locally through a user-friendly interface:
    ```shell
    docker run -d --network=host -v open-webui:/app/backend/data -e PORT=8090 -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui ghcr.io/open-webui/open-webui:main
    ```

   Once the Open WebUI container is running, open your browser and go to: http://localhost:8090.

#### Run the FastAPI Server
Simply run:
   ```shell
   poetry run uvicorn data_discovery_ai.server:app --reload --log-config=log_config.yaml
   ```
#### Running Tests
Run all tests with:
```shell
poetry run python -m unittest discover -s tests
```
#### Code formatting
Run manual checks:
```shell
pre-commit run --all-files
```
Checks are also executed when you run `git commit`. The configurations for pre-commit hooks are defined in `.pre-commit-config.yaml`.

#### Git Commit Guide (Optional)

We are using [gitmoji](https://gitmoji.dev/)(OPTIONAL) with husky and commitlint. Here you have an example of the most used ones:

- :art: - Improving structure/format of the code.
- :zap: - Improving performance.
- :fire: - Removing code or files.
- :bug: - Fixing a bug.
- :ambulance: - Critical hotfix.
- :sparkles: - Introducing new features.
- :memo: - Adding or updating documentation.
- :rocket: - Deploying stuff.
- :lipstick: - Updating the UI and style files.
- :tada: - Beginning a project.

Example of use:
`:wrench: add husky and commitlint config`

#### Branching name

- `hotfix/`: for quickly fixing critical issues,
- `usually/`: with a temporary solution
- `bugfix/`: for fixing a bug
- `feature/`: for adding, removing or modifying a feature
- `test/`: for experimenting something which is not an issue
- `wip/`: for a work in progress

And add the issue id after an `/` followed with an explanation of the task.

Example of use:
`feature/5348-create-react-app`

## ▶️ Using the API
Once the app is running, two routes are available:

| **Route**                        | **Description**                                                   |
|----------------------------------|-------------------------------------------------------------------|
| `GET /api/v1/ml/health`          | Health check                                                      |
| `POST /api/v1/ml/process_record` | One single point for calling AI models to process metadata record |
### Example Request Body
```JSON
{
    "selected_model":["description_formatting"],
    "title": "test title",
    "abstract": "test abstract"
}
```

**Required Header**
```shell
X-API-Key: your_api_key
```

(Must match the value of `API_KEY` specified in the environment variables).

**AI Model Options**

- `selected_model`: the AI models provided by `data-discovery-ai`. It should be a list of strings, which are the name of the AI task agents. Currently, four AI task agents available for distinctive tasks:
    - `keyword_classification`: predict keywords from AODN vocabularies based on metadata `title` and `abstract` with pretrained ML model.
    - `delivery_classification`: predict data delivery mode based on metadata `title`, `abstract`, and `lineage` with pretrained ML model.
    - `description_formatting`: reformatting long abstract into Markdown format based on metadata `title` and `abstract` with LLM model "gpt-4o-mini".
    - `link_grouping`: categorising links into four groups: ["Python Notebook", "Document", "Data Access", "Other"] based on metadata `links`.

## 🤖 Model Training (Optional)
Currently, two machine learning pipelines are available for training and evaluating models:

- `keyword`: keyword classification model, which is a Sequential model for multi-label classification task
- `delivery`: data delivery classification model, which is a self-learning model for binary classification task

### How to Run
To run one of the pipelines (for example, the keyword one), you can use the following command in your terminal:
```shell
python -m data_discovery_ai.ml.pipeline --pipeline keyword --start_from_preprocess False --model_name experimental
```
You can also use a shorter version:
```shell
 python -m data_discovery_ai.ml.pipeline -p keyword -s False -n experimental
```

### When Should I Re-Train?
If the raw data has changed (e.g., updated, cleaned, or expanded), you are recommended to re-train the model using the latest data.
To do this, set:
```shell
--start_from_preprocess True
```

As mentioned in [Environment variables](#environment-variables), ElasticSearch endpoint and API key are required to be set up in `.env` file.

### What Happens When I Run a Pipeline?
Running a pipeline trains a Machine Learning model and saves several resource files for reuse — so you don’t have to reprocess data or retrain the model every time.

1. `delivery` pipeline

    Outputs are saved in: `data_discovery_ai/resources/DataDeliveryModeFilter/`

    | File Name                 | Description                                     |
    |---------------------------|-------------------------------------------------|
    | `filter_preprocessed.pkl` | Preprocessed data used for training and testing |
    | `development.pkl`         | Trained binary classification model             |
    | `development.pca.pkl`     | PCA model used for dimensionality reduction     |

2. `keyword` pipeline

    Outputs are saved in: `data_discovery_ai/resources/KeywordClassifier/`

    | File Name            | Description                                           |
    |----------------------|-------------------------------------------------------|
    | `keyword_sample.pkl` | Preprocessed data used for training and testing       |
    | `keyword_label.pkl`  | Mapping between labels and internal IDs               |
    | `development.keras`  | Trained Keras model file (name set by `--model_name`) |

These files are **generated automatically during pipeline training** and **saved automatically after pipeline running**. They are intended for reuse in subsequent runs to avoid retraining.

#### Accepted Model Names
The `--model_name` argument helps organise different versions of your model. Here's how they’re typically used:

| Name           | Purpose                                    | When to Use                                           |
|----------------|--------------------------------------------|-------------------------------------------------------|
| `development`  | Active model development                   | For testing and iterating on new ideas                |
| `experimental` | Try new techniques or tuning               | For exploring new features or architectures           |
| `benchmark`    | Compare against the current baseline model | When validating improvements over a previous version  |
| `staging`      | Pre-production readiness                   | When testing full integration before final deployment |
| `production`   | Final production model                     | Live version used in production APIs or systems       |

> 🛠 **Tip:** When working locally, use `--model_name experimental` to avoid overwriting files used in deployments.

#### Model Naming Guidelines
Each model name reflects a stage in the model lifecycle:

1. Development
    - Initial model design and prototyping
    - Reaches minimum performance targets with stable training

2. Experimental
    - Shows consistent performance improvements
    - Experiment logs and results are clearly documented

3. Benchmark
    - Outperforms the existing benchmark (usually a copy of the production model)
    - Validated using selected evaluation metrics

4. Staging
    - Successfully integrated with application components (e.g. APIs)
    - Ready for deployment, pending final checks

5. Production
    - Deployed in a live environment
    - Monitored continuously, supports user feedback and live data updates

#### Model Usage
In the configuration file `data_discovery_ai/common/parameters.yaml`, you can specify which model version each task should use. For example:
```yaml
model:
  delivery_classification:
    pretrained_model: development
```
This means the agent handling the `delivery_classification` task will use the `development` version of the model.

### Track Training Progress with MLflow
We use [MLflow](https://mlflow.org/) to track model training and performance over time (like hypermeters, accuracy, precision, etc.).

It is automatically started at the beginning of the pipeline. Once a pipeline starts, you can open the tracking dashboard in your browser: [http://127.0.0.1:53000](http://127.0.0.1:53000)

You can change the model's training settings (like how long it trains or how fast it learns) by editing the trainer section in the file: `data_discovery_ai/config/parameters.yaml`

## 🛠️ Required Configuration Files
| **File**                                   | **Description**                                       |
|--------------------------------------------|-------------------------------------------------------|
| `data_discovery_ai/common/constants.py`    | 	Shared constants                                     |
| `data_discovery_ai/commom/parameters.yaml` | Store parameter settings for ML models and AI agents. |

## 📁 Project Structure
```
data_discovery_ai/
├── config/             # Common utilities and shared configurations/constants used across modules│
├── core/               # Core logic of the application such as API routes
├── agents/             # Task-specific agent modules using ML/AI/rule-based tools
├── ml/                 # Machine learning models: training, inference, evaluation logic
├── utils/              # Utility functions and helper scripts for various tasks
├── resources/          # Stored assets such as pretrained models, sample datasets, and other resources required for model inference
├── notebooks/          # Jupyter notebooks
├── tests/              # Unit tests for validating core components
│   ├── agents
│   ├── ml
│   └── utils
├── server.py             # FastAPI application entry point
```
