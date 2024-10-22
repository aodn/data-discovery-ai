"""
The keyword classification model used to identify the potential keywords for non-categorised records.
X is the embedding of the description, Y is its keywords
"""
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
import pandas as pd
import ast
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K


import logging
from matplotlib import pyplot as plt
from datetime import datetime

import os
os.environ["TF_USE_LEGACY_KERAS"] ="1"

DATASET = "./output/AODN_description.tsv"
KEYWORDS_DS = "./output/AODN_parameter_vocabs.tsv"
TARGET_DS = "./output/keywords_target.tsv"
# VOCABS = ['AODN Organisation Vocabulary', 'AODN Instrument Vocabulary', 'AODN Discovery Parameter Vocabulary', 'AODN Platform Vocabulary', 'AODN Parameter Category Vocabulary']
VOCABS = ['AODN Discovery Parameter Vocabulary']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def get_class_weights(Y_train):
    label_frequency = np.sum(Y_train, axis=0)
    total_samples = Y_train.shape[0]
    epsilon = 1e-6
    label_weights = np.minimum(1, 1 / (label_frequency + epsilon))

    label_weight_dict = {i: label_weights[i] for i in range(len(label_weights))}
    return label_weight_dict
    # num_labels = Y_train.shape[1]
    # class_weights = {}
    # for i in range(num_labels):
    #     y_label = Y_train[:, i]
    #     weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_label), y=y_label)
    #     class_weights[i] = {0: weight[0], 1: weight[1]}
    # return class_weights


def baseline(X_train, Y_train, model):
    if model == 'KNN':
        baseline_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
        baseline_model.fit(X_train, Y_train)
    if model == 'DT':
        baseModel = DecisionTreeClassifier(random_state=42)
        baseline_model = MultiOutputClassifier(baseModel)
        baseline_model.fit(X_train, Y_train)
    # TODO: add more baseline models

    return baseline_model

# weighted binary crossentropy
def weighted_binary_crossentropy(class_weights):
    def loss(y_true, y_pred):
        total_loss = 0.0
        
        for i in range(len(class_weights)):
            if class_weights[i][1] - class_weights[i][0] > 20:
                weight_0 = class_weights[i][0] * 10
                weight_1 = class_weights[i][1] / 2
            else:
                weight_0 = class_weights[i][0]
                weight_1 = class_weights[i][1]
            loss_pos = -weight_1 * y_true[:, i] * K.log(y_pred[:, i] + K.epsilon())
            loss_neg = -weight_0 * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + K.epsilon())
            
            label_loss = loss_pos + loss_neg
            total_loss += label_loss
        return K.mean(total_loss, axis=-1)
    return loss

def focal_loss(gamma, alpha):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)
    return focal_loss_fixed



def keyword_model(X_train, Y_train, X_test, Y_test, class_weight, dim, n_labels):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    model = Sequential([
        Input(shape=(dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        # Dense(64, activation='relu'),
        # Dropout(0.3),
        Dense(n_labels, activation='sigmoid')
    ])

    # Adam(learning_rate=1e-3)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=focal_loss(gamma=2., alpha=0.7), metrics=['accuracy', 'precision', 'recall'])

    model.summary()

    epoch = 100
    batch_size = 32

    # # class_weights = calculate_class_weights(Y_data=Y_train)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, min_lr=1e-6)

    # class_weight=class_weight, 
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, class_weight=class_weight, validation_split=0.1, callbacks=[early_stopping, reduce_lr])

    model.save(f"./output/saved/{current_time}-trained-keyword-epoch{epoch}-batch{batch_size}.keras")

    model.evaluate(X_test, Y_test)
    return model, history


def evaluation(Y_test, predictions):
    accuracy = accuracy_score(Y_test, predictions)
    hammingloss = hamming_loss(Y_test, predictions)
    precision = precision_score(Y_test, predictions, average='micro')
    recall = recall_score(Y_test, predictions, average='micro')
    f1 = f1_score(Y_test, predictions, average='micro')
    jaccard  = jaccard_score(Y_test, predictions, average='samples')

    print(f' {precision:.4f} | {recall:.4f} | {f1:.4f} | {hammingloss:.4f} |')
    return {
        'precision': f'{precision:.4f}',
        'recall': f'{recall:.4f}',
        'f1': f'{f1:.4f}',
        'hammingloss': f'{hammingloss:.4f}',
        'Jaccard Index': f'{jaccard:.4f}',
        'accuracy': f'{accuracy:.4f}'
    }


def prediction(X, model, confidence, top_N):
    predictions = model.predict(X)
    predicted_labels = (predictions > confidence).astype(int)

    for i in range(predicted_labels.shape[0]):
        if predicted_labels[i].sum() == 0:
            top_indices = np.argsort(predictions[i])[-top_N:]
            predicted_labels[i][top_indices] = 1

    return predicted_labels

def replace_with_column_names(row, column_names):
    return (' | '.join([column_names[i] for i, value in enumerate(row) if value == 1]))

def get_predicted_keywords(prediction, labels, targetDS):
    target_predicted = pd.DataFrame(prediction, columns=labels)
    predicted_keywords = target_predicted.apply(lambda row: replace_with_column_names(row, labels), axis=1)
    targetDS.drop(columns=['embedding', 'keywords'],inplace=True)

    output = pd.concat([targetDS, predicted_keywords], axis=1)
    output.columns = ['id', 'title', 'description', 'keywords']
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    output.to_csv(f"./output/saved/{current_time}.csv")
    logger.info(f'Save prediction to path output/saved/{current_time}.csv')
