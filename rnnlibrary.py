import numpy as np
import pandas as pd

import rnnlibrary

import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import accuracy_score

def rnn_model(
    rnn_layer, input_dim, embedding_matrix, max_token, units, hidden_activation, 
    epochs, output_activation, optimizer, loss_func, check_point_name, use_es,
    train_data, test_data, train_label, test_label, path_to_save_model):

    model = Sequential()

    model.add(Embedding(
          input_dim=input_dim,
          output_dim=100,
          weights=[embedding_matrix],
          input_length=max_token,
          trainable=False
    ))

    if rnn_layer == 'GRU':

        for unit in units:
            model.add(GRU(units=unit, activation=hidden_activation, return_sequences=True))
            model.add(Dropout(0.1))

        model.add(Dense(units=64))

        model.add(GRU(units=4, activation=hidden_activation, return_sequences=False))

        model.add(Dense(units=train_label.shape[1], activation=output_activation))

    elif rnn_layer == 'LSTM':

        for unit in units:
            model.add(GRU(units=unit, activation=hidden_activation, return_sequences=True))
            model.add(Dropout(0.1))

        model.add(Dense(units=64))

        model.add(GRU(units=4, activation=hidden_activation, return_sequences=False))

        model.add(Dense(units=train_label.shape[1], activation=output_activation))

    else:
        raise ValueError('Invalid RNN layer type!!!')

    model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    es = EarlyStopping(patience=5)

    checkpoint_filepath = path_to_save_model + "BestModels/best_model_" + check_point_name + "checkpoint.h5"

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    )
    if use_es:
      model_history = model.fit(train_data, train_label, epochs=epochs, validation_split=0.1, callbacks=[es, model_checkpoint_callback])

    else:
      model_history = model.fit(train_data, train_label, epochs=epochs, validation_split=0.1, callbacks=[model_checkpoint_callback])


    _, accuracy = model.evaluate(test_data, test_label)

    return accuracy, model_history, model