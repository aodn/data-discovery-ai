[![Language: Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![CI](https://github.com/aodn/data-discovery-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/aodn/data-discovery-ai/actions/workflows/ci.yml)

# Data Discovery AI

## Environment variables

If you wish to run the application locally, navigate to the root directory of the project and create a `.env` file.

Open the `.env` file and add the following lines to for using this application:

```shell
API_KEY="your_actual_api_key_here"
OPENAI_API_KEY="your_actual_openai_api_key"
```
These variables are required for the application to function properly.

**Note:** If you are running the application from a deployment environment, you do not need to set `API_KEY` manually as it has been already configured in the cloud environment. You may not need an OPENAI_API_KEY if `description_formatting` model was not called.

If you are going to train the model, make sure these variables have been set up for connecting Elasticsearch:
```shell
ES_ENDPOINT="your_actual_elasticsearch_endpoint"
ES_API_KEY="your_actual_es_api_key"
```

## Run the API server with Docker

Simply run `./startServer.sh` to run the app, this will create a docker image and run the image for you.

Host will be `http://localhost:8000`.

## Run the API server for development

### Requirements

- Conda (recommended for creating a virtual environment)

1. Install Conda (if not already installed):

   Follow the instructions at [Conda Installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Create Conda virtual environment:

   ```shell
   conda env create -f environment.yml
   ```

### Dependencies management

Poetry is used for dependency management, the `pyproject.toml` file is what is the most important, it will orchestrate the project and its dependencies.

You can update the file `pyproject.toml` for adding/removing dependencies by using

```shell
poetry add <pypi-dependency-name> # e.g poetry add numpy
poetry remove <pypi-dependency-name> # e.g. poetry remove numpy
```

You might want to update the `poetry.lock` file after manually modifying `pyproject.toml` with `poetry lock` command. To update all dependencies, use `poetry update` command.

### Installation and Usage

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

3. Run the FastAPI server:

   ```shell
   poetry run uvicorn data_discovery_ai.server:app --reload --log-config=log_config.yaml
   ```

4. Run the tests:

   ```shell
   poetry run pytest
   ```

### Code formatting

The command below is for manual checks; checks are also executed when you run `git commit`.

The configurations for pre-commit hooks are defined in `.pre-commit-config.yaml`.

```shell
pre-commit run --all-files
```

# Usage
When the server is running, there are two available routers:
- Health check through `/api/v1/ml/health`. Get method, no request parameter required.
- One single point for calling AI models to process metadata record through `/api/v1/ml/process_record`. Post method. JSON format request required. For example:
```JSON
{
    "selected_model":["description_formatting"],
    "title": "test title",
    "abstract": "test abstract"
}
```

**Required Fields**

- `selected_model`: the AI models provided by `data-discovery-ai`. It should be a list of strings, which are the name of the AI task agents. Currently, four AI task agents available for distinctive tasks:
    - `keyword_classification`: predict keywords from AODN vocabularies based on metadata `title` and `abstract` with pretrained ML model.
    - `delivery_classification`: predict data delivery mode based on metadata `title`, `abstract`, and `lineage` with pretrained ML model.
    - `description_formatting`: reformatting long abstract into Markdown format based on metadata `title` and `abstract` with LLM model "gpt-4o-mini".
    - `link_grouping`: categorising links into four groups: ["Python Notebook", "Document", "Data Access", "Other"] based on metadata `links`.
- For selected models, their required fields are needed to be passed.
- Routes are protected with authorisation checks. This means that the **request header** must include the key `X-API-Key`, and its value must match the `API_KEY` specified in the environment variables.

# File Structure
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
│   └── utils
├── server.py             # FastAPI application entry point
```

## Required Configuration Files
1. Global constants file
File name `constants.py` saved under folder `data_discovery_ai/common`.

2. Parameter configuration file
File name `parameters.yaml` saved under folder `data_discovery_ai/commom`. Store parameter settings for ML models and AI agents.


## Test
All test files are located in the `tests` folder at the root of the project. To run them, use the following command:

```bash
poetry run python -m unittest discover -s tests
```


## Commit

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

## Branching name

- `hotfix/`: for quickly fixing critical issues,
- `usually/`: with a temporary solution
- `bugfix/`: for fixing a bug
- `feature/`: for adding, removing or modifying a feature
- `test/`: for experimenting something which is not an issue
- `wip/`: for a work in progress

And add the issue id after an `/` followed with an explanation of the task.

Example of use:
`feature/5348-create-react-app`
