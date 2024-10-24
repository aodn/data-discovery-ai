import logging
import pickle
import pandas as pd
import ast
import os
import numpy as np
import json

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_to_file(obj, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(obj, file)
        logger.info(f"Saved to {file}")


def load_from_file(file_name):
    with open(file_name, "rb") as file:
        obj = pickle.load(file)
        logger.info(f"Load from {file}")
    return obj


"""
    Identify sample set from raw data.
    Input: 
        raw_data: type dataframe, which is the search result from ElasticSearch
        vocabs: type list, a list of vocabularies names. predefined in common/parameters.json file.
    Output:
        sampleSet: type dataframe. The identified sample set
"""
def identify_sample(raw_data, vocabs):
    raw_data_cleaned = raw_data[["_id", "_source.title", "_source.description", "_source.themes"]]
    raw_data_cleaned.columns = ["id", "title", "description", "keywords"]

    sampleSet = raw_data_cleaned[
        raw_data_cleaned["keywords"].apply(
            lambda terms: any(
                any(vocab in k["title"] for vocab in vocabs) for k in terms
            )
        )
    ]

    sampleSet.to_csv("data_discovery_ai/input/keywords_sample.tsv", index=False, sep="\t")
    return sampleSet

"""
    Preprocess sample set data, including extract and reformat labels, and remove empty value records
    Input:
        sampleSet: type dataframe, the identified sample set
        vocabs: type list, a list of vocabulary names, the predefined vocabularies
    Output:
        cleaned_sampleSet: type dataframe, the cleaned sample set
"""
def sample_preprocessor(sampleSet, vocabs):
    sampleSet["keywords"] = sampleSet["keywords"].apply(lambda x: keywords_formatter(x, vocabs))

    list_lengths = sampleSet["keywords"].apply(len)
    empty_keywords_records_index = list_lengths[list_lengths == 0].index.tolist()
    empty_keywords_records = []
    for index in empty_keywords_records_index:
        empty_keywords_records.append(sampleSet.iloc[index]["id"])
    empty_keywords_records
    cleaned_sampleSet = sampleSet[~sampleSet["id"].isin(empty_keywords_records)]

    return cleaned_sampleSet

"""
    Prepare input X and output Y matrix.
    Input: 
        sampleSet: type dataframe, sample set
    Output: 
        X: type numpy ndarray, feature variables for items in the sample set
        Y: type numpy ndarray, target variables for items in the sample set
        Y_df: type dataframe, target variables for items in the sample set
        labels: type list, predefined keyword set.
"""
def prepare_X_Y(sampleSet):
    X = np.array(sampleSet["embedding"].tolist())
    Y_df = prepare_Y_matrix(sampleSet)
    labels = Y_df.columns.to_list()
    Y = Y_df.to_numpy()
    return X, Y, Y_df, labels

"""
    Identify rare labels under a threshold.
    Input:
        Y_df: type dataframe. the target variables for all items in the sample set.
        threshold: type int, the threshold for identifing rare labels, if the number of apperance is under this threshold, the label is considered as a rare label.
        labels: type list, the predefined label set which contains all labels
    Output:
        rare_label_index: the indexes of rare labels in Y
"""
def identify_rare_labels(Y_df, threshold, labels):
    label_distribution = Y_df.copy()
    label_distribution = label_distribution.sum()
    label_distribution.sort_values()
    label_distribution_df = label_distribution.to_frame(name="count")
    rare_labels = label_distribution_df[label_distribution_df["count"] <= threshold].index.to_list()
    rare_label_index = []
    for item in rare_labels:
        if item in labels:
            index_in_labels = labels.index(item)
            rare_label_index.append(index_in_labels)
    return rare_label_index


def get_description_embedding(text):
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", clean_up_tokenization_spaces=False
    )
    model = BertModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()


def calculate_embedding(ds):
    tqdm.pandas()
    ds["embedding"] = ds["description"].progress_apply(
        lambda x: get_description_embedding(x)
    )
    return ds


def load_sample():
    try:
        sampleDS = load_from_file("./output/keywords_sample.pkl")
    except Exception as e:
        logger.info(
            "Files not Found: Missing keywords_sample.pkl in output folder. Try function load_sample_from_source() instead"
        )
    return sampleDS


def load_sample_from_source():
    dataset = load_from_file("./output/AODN.pkl")
    dataset.columns = ["id", "title", "description", "embedding"]

    sampleDS = pd.read_csv("./output/keywords_sample.tsv", sep="\t")
    sampleDS = sampleDS.merge(dataset, on=["id", "title", "description"])
    save_to_file(sampleDS, "./output/keywords_sample.pkl")
    return sampleDS


def load_target():
    try:
        targetDS = load_from_file("./output/keywords_target.pkl")
    except Exception as e:
        logger.info(
            "Files not Found: Missing keywords_target.pkl in output folder. Try function load_target_from_source() instead"
        )
    return targetDS


def extract_labels(ds, vocabs):
    ds["keywords"] = ds["keywords"].apply(lambda x: keywords_formatter(x, vocabs))
    return ds


def prepare_Y_matrix(ds):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(ds["keywords"])
    K = pd.DataFrame(Y, columns=mlb.classes_)

    if "" in K.columns:
        K.drop(columns=[""], inplace=True)
    return K


def keywords_formatter(text, vocabs):
    if type(text) is list:
        keywords = text
    else:
        keywords = ast.literal_eval(text)
    k_list = []
    for keyword in keywords:
        for concept in keyword["concepts"]:
            if keyword["title"] in vocabs and concept["id"] != "":
                concept_str = keyword["title"] + ":" + concept["id"]
                k_list.append(concept_str)
    return list(set(k_list))


def prepare_train_validation_test(X, Y, params):
    # get X, Y shape
    n_labels = Y.shape[1]
    dim = X[0].shape[0]

    n_splits = params["preprocessor"]["n_splits"]
    test_size = params["preprocessor"]["test_size"]
    train_test_random_state = params["preprocessor"]["train_test_random_state"]
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=train_test_random_state)

    for train_index, test_index in msss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

    print(f"Total samples: {len(X)}")
    print(f"Dimension: {dim}")
    print(f"No. of labels: {n_labels}")
    print(
        f"Train set size: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.2f}%)"
    )
    print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.2f}%)")

    return dim, n_labels, X_train, Y_train, X_test, Y_test


def customized_resample(X_train, Y_train, rare_class):
    X_augmented = X_train.copy()
    Y_augmented = Y_train.copy()
    num_copies = 10
    for label_idx in rare_class:
        sample_idx = np.where(Y_train[:, label_idx] == 1)[0]

        if len(sample_idx) == 1:
            sample_to_duplicate = sample_idx[0]
            for _ in range(num_copies):
                X_augmented = np.vstack([X_augmented, X_train[sample_to_duplicate]])
                Y_augmented = np.vstack([Y_augmented, Y_train[sample_to_duplicate]])
    return X_augmented, Y_augmented


def resampling(X_train, Y_train, strategy, rare_keyword_index):
    Y_train_combined = np.array(["".join(row.astype(str)) for row in Y_train])
    if strategy == "custom":
        X_train_resampled, Y_train_resampled = customized_resample(
            X_train, Y_train, rare_keyword_index
        )
    else:
        if strategy == "ROS":
            resampler = RandomOverSampler(sampling_strategy="auto", random_state=32)
        elif strategy == "RUS":
            resampler = RandomUnderSampler(
                sampling_strategy="auto", random_state=32
            )
        elif strategy == "SMOTE":
            resampler = SMOTE(k_neighbors=1, random_state=42)

        X_train_resampled, Y_combined_resampled = resampler.fit_resample(
            X_train, Y_train_combined
        )
        Y_train_resampled = np.array(
            [list(map(int, list(row))) for row in Y_combined_resampled]
        )
    
    print(" ======== After Resampling ========")
    print(f"Total samples: {len(X_train_resampled)}")
    print(f"Dimension: {X_train_resampled.shape[1]}")
    print(f"No. of labels: {Y_train_resampled.shape[1]}")
    print(f"X resampled set size: {X_train_resampled.shape[0]}")
    print(f"Y resampled set size: {Y_train_resampled.shape[0]}")
    return X_train_resampled, Y_train_resampled
