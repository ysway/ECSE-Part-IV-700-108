from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from tensorflow import keras
from pandas import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
import csv
import os

# Convert time series data to supervised data
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# data difference operation
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# data features, difference, scale, split
def prepare_data(series, n_test, n_lag, n_seq):
    # extract data from text
    raw_values = series.values
    # calculate data difference
    diff_series = difference(raw_values, 1)
    # obtian data aftere difference step
    diff_values = diff_series.values
    print(diff_values)
    # re-construct data to n rows and 1 column
    diff_values = diff_values.reshape(len(diff_values), 1)
    print(diff_values)
    # define the data scaler is between (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    print(scaler)
    # scale the data
    scaled_values = scaler.fit_transform(diff_values)
    print(scaled_values)
    # re-construct data to n rows and 1 column after scaling
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    print(scaled_values)
    # transfer data to a supervised data with step of n_sep
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    print(supervised)
    supervised_values = supervised.values
    # use number of n_test data for verification
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

# fitting a LSTM network, train data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # for every four sequences, the first element is input x, and last thress elements are predicting value y
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    # re-construct training data structure->[samples, timesteps, features]->[22,1,1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    print(X)
    print(y)
    # network structure
    model = Sequential()
    # a neuron， batch_input_shape(1,1,1)，propagate the status of sequence
    model.add(LSTM(n_neurons, batch_input_shape=(
        n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # start training
    for i in range(nb_epoch):
        # train data once with one unshuffled set of data
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        # reset the states of network for each training
        model.reset_states()
    return model

# LSTM unit step forecasting
def forecast_lstm(model, X, n_batch):
    # reshape the input (1,1,1) [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # forcast the shape of tensor as (1,3)
    forecast = model.predict(X, batch_size=n_batch)
    # return the result [[XX,XX,XX]] tp list
    return [x for x in forecast[0, :]]

# use model to predict
def make_forecasts(model, n_batch, test, n_lag, n_seq):
    forecasts = list()
    # check x value on-by-one
    for i in range(len(test)):
        # X, y = test[i, 0:n_lag], test[i, n_lag:]
        X = test[i, 0:n_lag]
        # LSTM unit forecasting
        forecast = forecast_lstm(model, X, n_batch)
        # store forecasted data
        forecasts.append(forecast)
    return forecasts

# inverse difference to forecasted data
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# inverse transform to forecasted data
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # inverse scale to data
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # inverse difference to data
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store transformed data
        inverted.append(inv_diff)
    return inverted

# evaluate forecasted data to root mean square
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        print('t+%d\r\n\tRMSE:\t%f\r\n\tR2:\t%f' % ((i+1), rmse, r2))

# plot
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()


#!#!#!#!#!#!#Program Entry#!#!#!#!#!#!#
# VA Extraction Begin
a = []
v = []

# RECOLA
inputPath = '../inputFile/emotional_behaviour/'
print('RECOLA Reading starts')
for i in range(16, 66):
    try:
        # arousal extraction
        with open(inputPath+'arousal/P'+str(i)+'.csv', 'r', newline='') as csv_file:
            reader = csv.reader(line.replace(';', ',') for line in csv_file)
            tVSa = list(reader)
        csv_file.close()
        tVSa = tVSa[1:]

        for index, data in enumerate(tVSa):
            sumA = 0
            for j in range(1, 7):
                sumA += float(data[j])

            a.append(sumA / 6.0)

        # valence extraction
        with open(inputPath+'valence/P'+str(i)+'.csv', 'r', newline='') as csv_file:
            reader = csv.reader(line.replace(';', ',') for line in csv_file)
            tVSv = list(reader)
        csv_file.close()

        tVSv = tVSv[1:]

        for index, data in enumerate(tVSv):
            sumA = 0
            for j in range(1, 7):
                sumA += float(data[j])

            v.append(sumA / 6.0)
    except:
        print('P'+str(i)+'.csv is missing, skipping...')

# JLCorpus
inputPath = '../inputFile/JLCorpus/all_speakers_arousal_valence.csv'
print('\r\nJLCorpus Reading starts')

# va extraction
with open(inputPath, 'r', newline='') as csv_file:
    reader = csv.reader(csv_file)
    tVSva = list(reader)
csv_file.close()

tVSva = tVSva[1:]

for data in tVSva:
    v.append(float(data[2]))
    a.append(float(data[3]))

# Semaine
print('\r\nSemaine Reading starts')
def vaArrayExtractor(inputPathV, inputPathA):
    # arousal extraction
    with open(inputPathA, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(' ', ',') for line in csv_file)
        tVSa = list(reader)
    csv_file.close()

    a = []
    for data in tVSa:
        if len(data)==0:
            break
        a.append(float(data[1]))
    
    # valence extraction
    with open(inputPathV, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(' ', ',') for line in csv_file)
        tVSv = list(reader)
    csv_file.close()
    
    v = []
    for data in tVSv:
        if len(data)==0:
            break
        v.append(float(data[1]))

    # VA length handler
    if len(v)<len(a):
        a = a[:len(v)]
    elif len(v)>len(a):
        v = v[:len(a)]
    else:
        pass

    return v, a


validSessionsText = open('./Semaine/validSessions.txt', 'r').readlines()
validSessions = [session[:-1] for session in validSessionsText[:-1]]
validSessions.append(validSessionsText[-1])
del validSessionsText

inputPath = '../inputFile/Sessions/'
for session in validSessions:
    print('Session: '+session)
    for f_name in os.listdir(inputPath+session+'/'):
        if f_name.endswith('V.txt'):
            try:
                tmpList = vaArrayExtractor(inputPath+session+'/'+f_name, inputPath+session+'/'+f_name[:-5]+'A.txt')
                v += tmpList[0]
                a += tmpList[1]
            except:
                print(
                    '\tPartial file ('+f_name[:-5]+') is missing, skipping...')

print('All Readings are finished\r\n')

'''
Number of data: 7929297
Prime Factorisation: (3**2)*347*2539
'''
# convert data to short save memory
v = (np.array(v)*1000).astype(dtype='int16',copy=True, errors='raise')
a = (np.array(a)*1000).astype(dtype='int16',copy=True, errors='raise')

# load data
series = Series(v, index = a)
del v, a, tVSa, tVSv, tVSva, reader, tmpList, validSessions
print(series)
# parameter setup
n_lag = 850       # use n_lag data
n_seq = 191       # forecast n_seq data
n_test = 7617    # test data is in n_test groups
n_epochs = 1    # train n_epochs times
n_batch = 1     # train n_batch data a time
n_neurons = 1   # number of neuron knots is n_neurons
# difference, scale, reshape data to unsupervised format
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fitting model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# start prediction
forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq)
# inverse transform forecasted data
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
# split the actual value which is corresponded to y from the training data
actual = [row[n_lag:] for row in test]
# inverse transform te actual value
actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate the rmse of actual and forecasted values
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot the result
plot_forecasts(series, forecasts, n_test+2)
