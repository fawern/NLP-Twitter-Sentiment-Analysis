from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def rnn_model(
    rnn_layer, input_dim, embedding_matrix, max_token, units, hidden_activation,
    epochs, output_activation, optimizer, check_point_name, use_es,
    train_data, test_data, train_label, test_label, path_to_save_model):

    model = Sequential()

    model.add(Embedding(
          input_dim=input_dim,
          output_dim=100,
          weights=[embedding_matrix],
          input_length=max_token,
          mask_zero=True,
          trainable=True
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

    try:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    except:
        raise ValueError('Invalid optimizer!!!')

    print(model.summary())

    es = EarlyStopping(patience=5)

    checkpoint_filepath = path_to_save_model + "BestModels/" + check_point_name
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