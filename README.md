[![Language: Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![CI](https://github.com/aodn/data-discovery-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/aodn/data-discovery-ai/actions/workflows/ci.yml)

# Data Discovery AI

## Environment variables

In the root directory of the project, create a `.env` file.

Open the `.env` file and add the following line to include your API key:

```shell
API_KEY=your_actual_api_key_here
```

## Run the API server with Docker

1. Build the Docker Image:

   ```shell
   docker compose build
   ```

2. Run the Services:

   ```shell
   docker compose up # [-d] for demon mode
   ```

3. Stop the Services

   ```shell
   docker compose down
   ```

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
    poetry run uvicorn data_discovery_ai.server:app --reload
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
