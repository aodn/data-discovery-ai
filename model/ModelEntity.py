import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, jaccard_score
from datetime import datetime

class BaseModel():
    """
        Base Model for multi-label classification tasks: keywords, parameters, organisation
    """
    def __init__(self, model=None):
        self.model = model

    def compile_model(self, optimizer, loss, metrics):
        if self.model:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit_model(self, X_train, Y_train, epochs, batch_size, validation_split, callbacks, class_weight=None):
        if self.model:
            history = self.model.fit(
                X_train, Y_train, epochs=epochs, batch_size=batch_size,
                validation_split=validation_split, callbacks=callbacks, class_weight=class_weight
            )
            return history

    def evaluate_model(self, X_test, Y_test):
        if self.model:
            return self.model.evaluate(X_test, Y_test)

    def save_model(self, filepath):
        if self.model:
            self.model.save(filepath)


class KeywordModel(BaseModel):
    def __init__(self, dim, n_labels):
        super().__init__()
        self.dim = dim
        self.n_labels = n_labels
        self.build_model()

    def build_model(self):
        self.model = Sequential([
            Input(shape=(self.dim,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.n_labels, activation='sigmoid')
        ])

    def train(self, X_train, Y_train, X_test, Y_test, class_weight=None, epochs=100, batch_size=32):
        self.compile_model(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy',
                           metrics=['accuracy', 'precision', 'recall', AUC()])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, min_lr=1e-6)

        history = self.fit_model(
            X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
            callbacks=[early_stopping, reduce_lr], class_weight=class_weight
        )

        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        filepath = f"./output/saved/{current_time}-trained-keyword-epoch{epochs}-batch{batch_size}.keras"
        
        self.save_model(filepath)
        return history

    @staticmethod
    def evaluation(Y_test, predictions):
        accuracy = accuracy_score(Y_test, predictions)
        hammingloss = hamming_loss(Y_test, predictions)
        precision = precision_score(Y_test, predictions, average='micro')
        recall = recall_score(Y_test, predictions, average='micro')
        f1 = f1_score(Y_test, predictions, average='micro')
        jaccard = jaccard_score(Y_test, predictions, average='samples')

        return {
            'accuracy': accuracy,
            'hammingloss': hammingloss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'Jaccard Index': jaccard
        }

    def predict_and_save(self, ds, confidence, labels):
        X = np.array(ds['embedding'].tolist())
        predictions = self.model.predict(X)
        predicted_labels = (predictions > confidence).astype(int)

        # Get label details
        predicted_keywords = []
        for i in range(len(predicted_labels)):
            lab = np.where(predicted_labels[i] == 1)[0]
            keywords = [labels[l] for l in lab]
            if len(keywords) == 0:
                predicted_keywords.append(None)
            else:
                predicted_keywords.append(' | '.join(keywords))
                
        ds['keywords'] = predicted_keywords
        ds.drop(columns=['embedding'], inplace=True)

        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        filepath = f"./output/saved/{current_time}.csv"
        ds.to_csv(filepath)
        return ds