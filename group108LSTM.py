from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import cross_val_score
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime

def timeTag():
    # YYmmddHHMM
    return datetime.now().strftime('[%Y%m%d%H%M]')

tTag = timeTag()

# prepare data for lstms
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
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
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Define scaler, feature number and number of step looking back
scale_range = (0, 1)
scaler = MinMaxScaler(feature_range=scale_range)
n_steps = 24  # exclude the current step
n_features = 7

usingJL = False
transformTarget = True

# read datasets, first n_steps data will be skipped
# Possible columns: Time,Valence,Arousal,RMS,F0,MFCC1,MFCC2,MFCC3,MFCC4,MFCC5,FileName,voiceTag
if usingJL:
    trainingDataset = pd.read_csv('inputFile/modelInput/allFileCombineP.csv')
    targetOfTrainingDataset = trainingDataset['Arousal'][n_steps:]
    trainingDataset = trainingDataset[['RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5']]
    print(trainingDataset.head(5))

    testingDataset = pd.read_csv('inputFile/modelInput/jlco0000st.csv')
    targetOfTestingDatasest = testingDataset['Arousal'][n_steps:]
    testingDataset = testingDataset[['RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5']]
    print(trainingDataset.head(5))
else:
    trainingDataset = pd.read_csv('inputFile/modelInput/allFileCombineP.csv')
    targetOfTrainingDataset = trainingDataset['Arousal'][n_steps:]
    trainingDataset = trainingDataset[['RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5']]
    print(trainingDataset.head(5))

if usingJL:
    # load and build training dataset
    values = trainingDataset.values
    # normalize features
    trainingScaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(trainingScaled, n_steps, 1)
    print(reframed.shape)
    values = reframed.values
    train = values

    # load and build testing dataset
    values = testingDataset.values
    # normalize features
    testingScaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(testingScaled, n_steps, 1)
    print(reframed.shape)
    values = reframed.values
    test = values
else:
    values = trainingDataset.values
    # normalize features
    trainingScaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(trainingScaled, n_steps, 1)
    print(reframed.shape)
    # split into train and test sets
    values = reframed.values
    n_train_steps = 1650780  # 90% of dataset, total length: 1834200
    train = values[:n_train_steps, :]
    test = values[n_train_steps:, :]

# transforming targets
if transformTarget:
    trainingyScaled = scaler.fit_transform(np.array(targetOfTrainingDataset).reshape(-1, 1))

    # seems no need to scale the test y, as it is only used for comparison
    # testingyScaled = scaler.fit_transform(np.array(targetOfTestingDatasest).reshape(-1, 1))

# split into input and outputs
if usingJL:
    # n_obs = (n_steps + 1) * n_features
    # train_X, train_y = train[:, :n_obs], targetOfTrainingDataset
    # test_X, test_y = test[:, :n_obs], targetOfTestingDatasest
    train_X = train
    train_y = trainingyScaled[:, 0]

    test_X = test
    test_y = targetOfTestingDatasest
else:
    n_obs = n_steps * n_features
    if transformTarget:
        train_X, train_y = train, trainingyScaled[:n_train_steps, 0]
    else:
        train_X, train_y = train, targetOfTrainingDataset[:n_train_steps]
    test_X, test_y = test, targetOfTrainingDataset[n_train_steps:]
    print(train_X.shape, train_y.shape)

# reshape input to be 3D [samples, timesteps (n_steps before + 1 current step), features]
train_X = train_X.reshape((train_X.shape[0], n_steps + 1, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps + 1, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def create_model():
    model = keras.Sequential([
        layers.LSTM(49, input_shape=(train_X.shape[1], train_X.shape[2])),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss='mse', metrics=['accuracy'])

    return model

# Create a KerasClassifier with best parameters
model = KerasRegressor(build_fn=create_model, batch_size=225, epochs=35, shuffle=False)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, train_X, train_y, cv=7, scoring='r2')

# Get the accuracy
print('The mean accuracy:', kfolds.mean())

# use callbacks
checkpoint = ModelCheckpoint("", monitor="val_loss", verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto', restore_best_weights=True)

# fit network [3, 3, 5, 5, 3433]
history = model.fit(train_X, train_y, epochs=50, batch_size=75, validation_split=0.2, verbose=2, shuffle=False, callbacks=[early_stop, checkpoint, reduce_lr])

print(history.history.keys())

# plot history
# loss
plt.ioff()
fig = plt.figure(figsize=[48, 48])
fig.suptitle('Loss Comparison', fontsize=16)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend(loc='upper right')
plt.savefig('outputFile/ModelPlots/'+tTag+'trainLossVsVal.png', format='png')
plt.close(fig)

plt.ioff()
fig = plt.figure(figsize=[48, 48])
fig.suptitle('Accuracy Comparison', fontsize=16)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend(loc='upper left')
plt.savefig('outputFile/ModelPlots/'+tTag+'trainAccuracyVsVal.png', format='png')
plt.close(fig)

# save the model
model.model.save('outputFile/Models/'+tTag+'EsModel')
# https://stackoverflow.com/questions/42666046/loading-a-trained-keras-model-and-continue-training

# make a prediction
if transformTarget:
    inv_yPredict = model.predict(test_X)
    # inv transform the predicted value
    yPredict = scaler.inverse_transform(inv_yPredict.reshape(-1, 1))
else:
    yPredict = model.predict(test_X)
yPredict = yPredict[:, 0]

# actual value
yActual = test_y
# calculate RMSE
rmse = np.sqrt(mean_squared_error(yActual, yPredict))
print('Test RMSE: %.3f' % rmse)

r2_score(yActual, yPredict)

plt.ioff()
fig = plt.figure(figsize=[48, 48])
fig.suptitle('Actual vs Prediction', fontsize=16)
pred_test_list = [i for i in yPredict]
submission = pd.DataFrame({'Arousal': yActual, 'Prediction': pred_test_list})
submission.loc[1:, ['Arousal', 'Prediction']].plot()
plt.savefig('outputFile/ModelPlots/'+tTag+'actualVsPrediction.png', format='png')
plt.close(fig)
submission.to_csv('outputFile/Submissions/'+tTag+'es2jSubmission.csv', index=False)

# print(pred_test_list[1000:1150])

correlation = submission.corr(method='pearson')
print(correlation)

d0 = submission[['Arousal', 'Prediction']]
plt.ioff()
fig = plt.figure(figsize=[48, 48])
fig.suptitle('Actual Prediction Correlation', fontsize=16)
sns.pairplot(d0, kind="scatter")
plt.savefig('outputFile/ModelPlots/'+tTag+'actualPredictionCorrelation.png', format='png')
plt.close(fig)
