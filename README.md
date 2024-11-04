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

### Versioning

This project uses **semantic versioning** with automated releases managed by `semantic-release`.

Every code change with commits following [Conventional Commits](https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional) will trigger a version update and create a GitHub release.

**Commit Guidelines**

- `feat:` For new features
- `fix:` For bug fixes
- `BREAKING CHANGE:` For any breaking changes


# Edge/syste/prod

models name stricly controller:

available options : `development`, `staging`, `production`, `test`

# Devlopment

syntax is :/....


# File Structure
## Required Configuration Files
1. Elasticsearch configuration file
File name `esManager.ini` saved under folder `data_discovery_ai/common`. Specific fileds & values required:
   1. `end_point`: the Elasticsearch endpoint of a deployment
   2. `api_key`: the API key used for access Elasticsearch
2. Keyword classification parameter configuration file
File name `keyword_classification_parameters.ini` saved under folder `data_discovery_ai/common`. Required two sections: `preprocessor` to set up parameters used for data preprocessing module, and `keywordModel` to set up parameters used for training and evaluation of the keyword model. Here are the definitions of fields:
   1. `preprocessor`

   | Parameter | Definition | Default Value used |
   | ---- | ---- | ---- |
   | vocabs | Titles of vocabularies used to identify samples from raw data; multiple values can be separated by ', '. | AODN Instrument Vocabulary, AODN Discovery Parameter Vocabulary, AODN Platform Vocabulary |
   | rare_label_threshold | The threshold for identifying a rare label, defined as the number of occurrences of the label across all sample records, should be an integer. | 10 |
   | test_size | A floating-point number in the range [0, 1], indicating the percentage of the test set size relative to all samples. | 0.2 |
   | n_splits | Number of re-shuffling & splitting iterations for cross validation, used as the value of parameter `n_splits` when initialise an object of `MultilabelStratifiedShuffleSplit`. | 5 |
   | train_test_random_state | The seed for splitting the train and test sets, used as the value of the `random_state` parameter when initialising an instance of `MultilabelStratifiedShuffleSplit`. | 42 |

   2. `keywordModel`

   | Parameter | Definition | Defalt Value used |
   | ---- | ---- | ---- |
   | dropout | The probability of a neuron being dropped. A strategy used for avoiding overfitting. | 0.3 |
   | learning_rate | A hyperparameter determines how much the model's parameters are adjusted with respect to the gradient of the loss function. | 0.001 |
   | fl_gamma | The $\gamma$ parameter of the focal loss function, which adjusts the focus of the loss function on hard-to-classify samples. It should be an integer. | 2 |
   | fl_alpha | The $\alpha$ parameter of the focal loss function, which balances the importance of positive and negative samples. It should be a floating-point number between 0 and 1. | 0.7 |
   | epoch | The number of times the train set is passed through the model for training. It should be an integer. | 100 |
   | batch | The batch size which defines the number of samples in each batch. | 32 |
   | validation_split | The percentage of the training set to be used as the validation set. | 0.2 |
   | confidence | The probability threshold for identifying a label as positive (value 1). | 0.5 |
   | top_N | The number of labels to select using argmax(probability) if no labels reach the confidence threshold. | 2 |
