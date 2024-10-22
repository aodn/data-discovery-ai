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

DATASET = "./output/AODN.tsv"
KEYWORDS_DS = "./output/keywords_sample.tsv"
TARGET_DS = "./output/keywords_target.tsv"
VOCABS = ['AODN Instrument Vocabulary', 'AODN Discovery Parameter Vocabulary', 'AODN Platform Vocabulary']
# VOCABS = ['AODN Discovery Parameter Vocabulary']


def save_to_file(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)
        logger.info(f'Saved to {file}')

def load_from_file(file_name):
    with open(file_name, 'rb') as file:
        obj = pickle.load(file)
        logger.info(f'Load from {file}')
    return obj


def identify_sample(vocabs):
    ds = pd.read_csv(DATASET, sep='\t')
    keywords = ds[['_id', '_source.title', '_source.description',  '_source.themes']]
    keywords.columns = ['id', 'title', 'description', 'keywords']
    keywords.loc[:, 'keywords'] = keywords['keywords'].apply(lambda k:eval(k))

    sample = keywords[keywords['keywords'].apply(
        lambda terms: any(any(vocab in k['title'] for vocab in vocabs) for k in terms)
    )]
    
    sample.to_csv("./output/keywords_sample.tsv", index=False, sep='\t')
    return sample



def get_description_embedding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :] 
    return cls_embedding.squeeze().numpy()


def calculate_embedding(ds):
    tqdm.pandas()
    ds['embedding'] = ds['description'].progress_apply(lambda x: get_description_embedding(x))
    return ds

def load_sample():
    try:
        sampleDS = load_from_file('./output/keywords_sample.pkl')
    except Exception as e:
        logger.info("Files not Found: Missing keywords_sample.pkl in output folder. Try function load_sample_from_source() instead")  
    return sampleDS


def load_sample_from_source():
    dataset = load_from_file('./output/AODN.pkl')
    dataset.columns = ['id', 'title', 'description', 'embedding']

    sampleDS = pd.read_csv('./output/keywords_sample.tsv', sep='\t')
    sampleDS = sampleDS.merge(dataset, on=['id', 'title', 'description'])
    save_to_file(sampleDS, './output/keywords_sample.pkl')
    return sampleDS

def load_target():
    try:
        targetDS = load_from_file('./output/keywords_target.pkl')
    except Exception as e:
        logger.info("Files not Found: Missing keywords_target.pkl in output folder. Try function load_target_from_source() instead")  
    return targetDS

def load_datasets(vocabs):
    try:
        targetDS = load_from_file('./output/keywords_target.pkl')
        sampleDS = load_from_file('./output/keywords_sample.pkl')
        preprocessed_keywordDS, labels_df = extract_labels(sampleDS, vocabs)

        # drop empty keyword column
        if '' in labels_df.columns:
            labels_df.drop(columns=[''], inplace=True)

    except Exception as e:
        logger.info("Files not Found: Missing keywords_target.pkl and keywords_sample.pkl in output folder.")
    return targetDS, preprocessed_keywordDS, labels_df

def load_datasets_from_source(vocabularies):
    source = pd.read_csv(DATASET, sep='\t')
    if not os.path.exists('./output/AODN.pkl'):
        dataset = source
        dataset = calculate_embedding(dataset)
        dataset.columns = ['id', 'title', 'description', 'embedding']
        save_to_file(dataset, './output/AODN.pkl')
    else:
        dataset = load_from_file('./output/AODN.pkl')
        dataset.columns = ['id', 'title', 'description', 'embedding']

    # load from file which has calculated embeddings to save computation time
    targetDS = pd.read_csv(TARGET_DS, sep='\t')
    targetDS = targetDS.merge(dataset, on=['id', 'title', 'description'])
    save_to_file(targetDS, './output/keywords_target.pkl')

    if not os.path.exists(KEYWORDS_DS):
        identify_sample(vocabularies)

    keywordDS = pd.read_csv(KEYWORDS_DS, sep='\t')
    keywordDS = keywordDS.merge(dataset, on=['id', 'title', 'description'])
    save_to_file(keywordDS, './output/keywords_sample.pkl')
    
    preprocessed_keywordDS = extract_labels(keywordDS, vocabularies)

    return dataset, targetDS, preprocessed_keywordDS


def extract_labels(ds, vocabs):
    ds['keywords'] = ds['keywords'].apply(lambda x: keywords_formatter(x, vocabs))
    # mlb = MultiLabelBinarizer()
    # Y = mlb.fit_transform(ds['keywords'])
    # labels_df = pd.DataFrame(Y, columns=mlb.classes_)
    # save_to_file(labels_df, './output/AODN_vocabs_label.pkl')
    # return ds, labels_df
    return ds

def prepare_Y_matrix(ds):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(ds['keywords'])
    K = pd.DataFrame(Y, columns=mlb.classes_)
    
    if '' in K.columns:
        K.drop(columns=[''], inplace=True)
    return K

def keywords_formatter(text, vocabs):
    if type(text) is list:
        keywords = text
    else:
        keywords = ast.literal_eval(text)
    k_list = []
    for keyword in keywords:
        for concept in keyword['concepts']:
            if keyword['title'] in VOCABS and concept['id'] != '':
                concept_str = keyword['title']  + ':' + concept['id']
                k_list.append(concept_str)
    return list(set(k_list))


def prepare_train_validation_test(X, Y):
    # get X, Y shape
    n_labels = Y.shape[1]
    # Y = Y.to_numpy()
    dim = X[0].shape[0]

    msss = MultilabelStratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for train_index, test_index in msss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    
    print(f"Total samples: {len(X)}")
    print(f"Dimension: {dim}")
    print(f"No. of labels: {n_labels}")
    print(f"Train set size: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.2f}%)")
    # print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0] / len(X) * 100:.2f}%)")
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
    print(f"Total samples: {len(X_train)}")
    print(f"Dimension: {X_train.shape[1]}")
    print(f"No. of labels: {X_train.shape[1]}")
    print(f"X set size: {X_train.shape[0]}")
    print(f"Y set size: {X_train.shape[0]}")
    Y_train_combined = np.array([''.join(row.astype(str)) for row in Y_train])
    if strategy == 'custom':
        X_train_resampled, Y_train_resampled = customized_resample(X_train, Y_train,rare_keyword_index)
    else:
        if strategy == 'ROS':
            resampler = RandomOverSampler(sampling_strategy='auto', random_state=32)
        elif strategy == 'RUS':
            resampler = RandomUnderSampler(sampling_strategy='majority', random_state=32)
        elif strategy == 'SMOTE':
            resampler = SMOTE(k_neighbors=1,random_state=42)

        X_train_resampled, Y_combined_resampled = resampler.fit_resample(X_train, Y_train_combined)
        Y_train_resampled = np.array([list(map(int, list(row))) for row in Y_combined_resampled])

    print(f"Total samples: {len(X_train_resampled)}")
    print(f"Dimension: {X_train_resampled.shape[1]}")
    print(f"No. of labels: {Y_train_resampled.shape[1]}")
    print(f"X resampled set size: {X_train_resampled.shape[0]}")
    print(f"Y resampled set size: {Y_train_resampled.shape[0]}")
    return X_train_resampled, Y_train_resampled