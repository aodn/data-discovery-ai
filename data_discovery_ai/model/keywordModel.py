"""
The keyword classification model used to identify the potential keywords for non-categorised records.
X is the embedding of the description, Y is its keywords
"""

import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)
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

os.environ["TF_USE_LEGACY_KERAS"] = "1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_class_weights(Y_train):
    label_frequency = np.sum(Y_train, axis=0)
    total_samples = Y_train.shape[0]
    epsilon = 1e-6
    label_weights = np.minimum(1, 1 / (label_frequency + epsilon))

    label_weight_dict = {i: label_weights[i] for i in range(len(label_weights))}
    return label_weight_dict


def baseline(X_train, Y_train, model):
    if model == "KNN":
        baseline_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
        baseline_model.fit(X_train, Y_train)
    if model == "DT":
        baseModel = DecisionTreeClassifier(random_state=42)
        baseline_model = MultiOutputClassifier(baseModel)
        baseline_model.fit(X_train, Y_train)
    # TODO: add more baseline models

    return baseline_model


def focal_loss(gamma, alpha):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = -alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)

    return focal_loss_fixed


def keyword_model(X_train, Y_train, X_test, Y_test, class_weight, dim, n_labels, params):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    model = Sequential(
        [
            Input(shape=(dim,)),
            Dense(128, activation="relu"),
            Dropout(params["keywordModel"]["dropout"]),
            # Dense(64, activation='relu'),
            # Dropout(params["keywordModel"]["dropout"]),
            Dense(n_labels, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=params["keywordModel"]["learning_rate"]),
        loss=focal_loss(
            gamma=params["keywordModel"]["fl_gamma"], alpha=params["keywordModel"]["fl_alpha"]
        ),
        metrics=["accuracy", "precision", "recall"],
    )

    model.summary()

    epoch = params["keywordModel"]["epoch"]
    batch_size = params["keywordModel"]["batch"]

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=params["keywordModel"]["early_stopping_patience"], restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=params["keywordModel"]["reduce_lr_patience"], min_lr=1e-6)

    history = model.fit(
        X_train,
        Y_train,
        epochs=epoch,
        batch_size=batch_size,
        class_weight=class_weight,
        validation_split=params["keywordModel"]["validation_split"],
        callbacks=[early_stopping, reduce_lr],
    )

    model.save(
        f"data_discovery_ai/output/{current_time}-trained-keyword-epoch{epoch}-batch{batch_size}.keras"
    )

    model.evaluate(X_test, Y_test)
    return model, history


def evaluation(Y_test, predictions):
    accuracy = accuracy_score(Y_test, predictions)
    hammingloss = hamming_loss(Y_test, predictions)
    precision = precision_score(Y_test, predictions, average="micro")
    recall = recall_score(Y_test, predictions, average="micro")
    f1 = f1_score(Y_test, predictions, average="micro")
    jaccard = jaccard_score(Y_test, predictions, average="samples")

    print(f" {precision:.4f} | {recall:.4f} | {f1:.4f} | {hammingloss:.4f} |")
    return {
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1": f"{f1:.4f}",
        "hammingloss": f"{hammingloss:.4f}",
        "Jaccard Index": f"{jaccard:.4f}",
        "accuracy": f"{accuracy:.4f}",
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
    return " | ".join([column_names[i] for i, value in enumerate(row) if value == 1])


def get_predicted_keywords(prediction, labels):
    target_predicted = pd.DataFrame(prediction, columns=labels)
    predicted_keywords = target_predicted.apply(
        lambda row: replace_with_column_names(row, labels), axis=1
    )
    return predicted_keywords
    # targetDS.drop(columns=["embedding", "keywords"], inplace=True)

    # output = pd.concat([targetDS, predicted_keywords], axis=1)
    # output.columns = ["id", "title", "description", "keywords"]
    # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # output.to_csv(f"./output/saved/{current_time}.csv")
    # logger.info(f"Save prediction to path output/saved/{current_time}.csv")
