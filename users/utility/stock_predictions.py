import time
import numpy as np
import pandas as pd
import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
from django.conf import settings

def parser(x):
    return datetime.datetime.strptime(x,'%m/%d/%Y')

# FEATURE GENERATION
def get_technical_indicators(dataset):  # function to generate feature technical indicators
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
    # Create MACD
    dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])
    # Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(window=20).std()
    dataset['upper_band'] = (dataset['Close'].rolling(window=20).mean()) + (dataset['20sd'] * 2)
    dataset['lower_band'] = (dataset['Close'].rolling(window=20).mean()) - (dataset['20sd'] * 2)
    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
    # Create Momentum
    dataset['momentum'] = (dataset['Close'] / 100) - 1
    return dataset


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0 - last_days
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['Close'], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Amazon - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['momentum'], label='Momentum', color='b', linestyle='-')
    plt.legend()
    plt.show()


def get_fourier(dataset):
    data_FT = dataset[['Date', 'Close']]
    close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
    close_fft = np.fft.ifft(close_fft)
    close_fft
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_list_m10= np.copy(fft_list); fft_list_m10[100:-100]=0
    dataset['Fourier'] = pd.DataFrame(fft_list_m10).apply(lambda x: np.abs(x))
    #dataset['absolute'] = dataset['Fourier'].apply(lambda x: np.abs(x))
    return dataset


def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['Close']
    X = data.iloc[:, 1:19]

    train_samples = int(X.shape[0] * 0.65)

    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)

def start_process():
    import datetime
    path = os.path.join(settings.MEDIA_ROOT, 'AMZN.csv')
    dataset_ex_df = pd.read_csv(path, header=0, parse_dates=[0], date_parser=parser)
    print(dataset_ex_df[['Date', 'Close']].head(3))
    print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(dataset_ex_df['Date'], dataset_ex_df['Close'], label='Amazon stock')
    plt.vlines(datetime.date(2016, 4, 20), 0, 270, linestyles='--', colors='gray', label='Train/Test data cut-off')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.title('Figure 2: Amazon stock price')
    plt.legend()
    plt.show()

    #  FEATURE GENERATION
    dataset_TI_df = get_technical_indicators(dataset_ex_df)
    print(dataset_TI_df.head())

    # Plot Technical
    plot_technical_indicators(dataset_TI_df, 400)

    data_FT = dataset_ex_df[['Date', 'Close']]
    close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 100]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(data_FT['Close'], label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 3: Amazon (close) stock prices & Fourier transforms')
    plt.legend()
    plt.show()

    dataset_TI_df = get_fourier(dataset_ex_df)
    print(dataset_TI_df.head(30))
    from collections import deque
    items = deque(np.asarray(fft_df['absolute'].tolist()))
    items.rotate(int(np.floor(len(fft_df) / 2)))
    plt.figure(figsize=(10, 7), dpi=80)
    plt.stem(items)
    plt.title('Figure 4: Components of Fourier transforms')
    plt.show()

    # Arima Model
    from statsmodels.tsa.arima_model import ARIMA
    from pandas import DataFrame
    from pandas import datetime

    series = data_FT['Close']
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(series)
    plt.figure(figsize=(10, 7), dpi=80)
    plt.show()

    from pandas import read_csv
    from pandas import datetime
    from pandas import DataFrame
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error

    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        #print(f'Future values {yhat} Data is {t}')
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    dataset_TI_df['ARIMA'] = pd.DataFrame(predictions)

    error = mean_squared_error(test, predictions)
    rmse = math.sqrt(error)
    print('Test MSE: %.3f' % error)

    # Plot the predicted (from ARIMA) and real prices

    # plt.figure(figsize=(12, 6), dpi=100)
    # plt.plot(test, color='black', label='Real')
    # plt.plot(predictions, color='yellow', label='Predicted')
    # plt.xlabel('Days')
    # plt.ylabel('USD')
    # plt.title('Figure 5: ARIMA model on Amazon stock')
    # plt.legend()
    # plt.show()

    print(dataset_ex_df.head(8))
    print('Total dataset has {} samples, and {} features.'.format(dataset_ex_df.shape[0], dataset_ex_df.shape[1]))
    # Get training and test data
    (X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df)
    regressor = xgb.XGBRegressor(gamma=0.0, n_estimators=200, base_score=0.7, colsample_bytree=1, learning_rate=0.05)
    xgbModel = regressor.fit(X_train_FI, y_train_FI, \
                             eval_set=[(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], \
                             verbose=False)
    eval_result = regressor.evals_result()
    training_rounds = range(len(eval_result['validation_0']['rmse']))
    plt.scatter(x=training_rounds, y=eval_result['validation_0']['rmse'], label='Training Error')
    plt.scatter(x=training_rounds, y=eval_result['validation_1']['rmse'], label='Validation Error')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Training Vs Validation Error')
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(),
            tick_label=X_test_FI.columns)
    plt.title('Figure 6: Feature importance of the technical indicators.')
    plt.show()

    # LSTM

    # 1. take dataframe and drop na
    dataset_lstm_df = dataset_TI_df.drop(columns='Date')
    dataset_lstm_df.head(7)#1. take dataframe and drop na
    dataset_lstm_df = dataset_TI_df.drop(columns='Date')
    print(dataset_lstm_df.head(7))
    print('Total dataset has {} samples, and {} features.'.format(dataset_lstm_df.shape[0],dataset_lstm_df.shape[1]))
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.layers import Flatten

    # creating test, train and validate trains
    train, validate, test = np.split(dataset_lstm_df.sample(frac=1), [int(.6 * len(dataset_lstm_df)), int(.8 * len(dataset_lstm_df))])
    open_training = train.iloc[:, 1:2].values
    # normalise
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    open_training = scaler.fit_transform(open_training)
    # convert to right shape
    features_set_1 = []
    labels_1 = []
    for i in range(60, 450):
        features_set_1.append(open_training[i - 60:i, 0])
        labels_1.append(open_training[i, 0])

    # Code ref: https://github.com/LiamConnell/deep-algotrading
    features_set_1, labels_1 = np.array(features_set_1), np.array(labels_1)
    features_set_1 = np.reshape(features_set_1, (features_set_1.shape[0], features_set_1.shape[1], 1))
    # training it
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set_1.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    model.fit(features_set_1, labels_1, epochs=100, batch_size=32, validation_data=(features_set_1, labels_1))

    # TESTING THE MODEL
    open_testing_processed = test.iloc[:, 1:2].values
    # convert test data to right format
    open_total = pd.concat((train['Open'], test['Open']), axis=0)
    test_inputs = open_total[len(open_total) - len(test) - 60:].values
    # scaling data
    test_inputs = test_inputs.reshape(-1, 1)
    test_inputs = scaler.transform(test_inputs)
    test_features = []
    for i in range(60, 151):
        test_features.append(test_inputs[i - 60:i, 0])
    test_features = np.array(test_features)
    test_features.shape
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    rslt_dict = {}
    # make predictions
    predictions = model.predict(test_features)
    predictions = scaler.inverse_transform(predictions)
    plt.figure(figsize=(10, 6))
    plt.plot(open_testing_processed, color='pink', label='Actual Stock Price')
    plt.plot(predictions, color='yellow', label='Predicted Stock Price')
    rslt_dict.update({'actual': open_testing_processed, 'predictions': predictions})
    plt.title('Actual Value vs Predicted in Futures')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.show()
    rslt_dict.update({'error': error})
    rslt_dict.update({'rmse': rmse})
    #rslt_dict = pd.DataFrame(rslt_dict)
    return rslt_dict





