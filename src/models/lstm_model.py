'''
LSTM Forecasting Model

This module performs:
1. Data Scaling
2. Sequence Generation
3. Hyperparameter Tuning
4. Model Training
5. Forecasting
6. Performance Evaluation
'''

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input

from evaluation import evaluate_model

# Create Sequences
def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# Build Model
def build_model(units):

    model = Sequential()
    sequence_length = 30

    model.add(Input(shape=(sequence_length, 1)))
    model.add(LSTM(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

# Hyperparameter Tuning

def tune_model(X_train, y_train, X_validation, y_validation):

    print('\nSearching Best LSTM Parameters...\n')

    best_rmse  = float('inf')
    best_model = None

    units_list  = [32, 50]
    batch_sizes = [16, 32]
    epochs_list = [30, 50]

    for units in units_list:
        for batch in batch_sizes:
            for epochs in epochs_list:
                model = build_model(units)

                early_stop = EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights = True)

                model.fit(
                    X_train,
                    y_train,
                    validation_data=(
                        X_validation,
                        y_validation
                    ),
                    epochs=epochs,
                    batch_size=batch,
                    callbacks=[early_stop],
                    verbose=0
                )

                predictions = model.predict(X_validation, verbose = 0)

                rmse = np.sqrt(np.mean((predictions.flatten() - y_validation)**2))

                print(
                    f'Units={units} | '
                    f'Batch={batch} | '
                    f'Epochs={epochs} '
                    f'--> RMSE={rmse:.4f}'
                )

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model

    print('\nBest Validation RMSE:', best_rmse)

    best_units = units
    best_batch = batch
    best_epochs = epochs

    return best_units, best_batch, best_epochs

# Main Wrapper
def run_lstm(train_df, validation_df, test_df):

    print('=' * 60)
    print('LSTM MODEL')
    print('=' * 60)

    scaler = MinMaxScaler()
    sequence_length = 60

    train_scaled      = scaler.fit_transform(train_df[['Close']])

    validation_input = pd.concat([train_df.tail(sequence_length), validation_df]) # last 30 days 
    validation_scaled = scaler.transform(validation_input[['Close']])

    test_input = pd.concat([validation_df.tail(sequence_length),test_df])
    test_scaled = scaler.transform(test_input[['Close']])     

    X_train, y_train           = create_sequences(train_scaled, sequence_length)
    X_validation, y_validation = create_sequences(validation_scaled, sequence_length)
    X_test, y_test             = create_sequences(test_scaled, sequence_length)

    X_train      = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)
    X_test       = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    best_units, best_batch, best_epochs = tune_model(X_train, y_train, X_validation, y_validation)

    # Retrain Final Model using Train + Validation
    train_validation_df = pd.concat([train_df, validation_df])
    train_validation_scaled = scaler.fit_transform(train_validation_df[['Close']])
    X_train_final, y_train_final = create_sequences(train_validation_scaled, sequence_length)
    X_train_final = X_train_final.reshape(X_train_final.shape[0], X_train_final.shape[1], 1)

    # Build a NEW model using the best hyperparameters
    model = build_model(best_units)
    early_stop = EarlyStopping(monitor='loss', patience = 5, restore_best_weights = True)
    model.fit(X_train_final, y_train_final,  epochs = best_epochs, batch_size = best_batch, callbacks = [early_stop], verbose = 0)

    # Test Prediction
    predictions = model.predict(X_test, verbose = 0)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    results = evaluate_model(actual = actual.flatten(), predicted = predictions.flatten(), model_name = 'LSTM')

    return results    
    '''
    predictions = model.predict(X_test, verbose = 0)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))  
    results = evaluate_model(actual = actual.flatten(), predicted = predictions.flatten(), model_name = 'LSTM')
    return results
    '''