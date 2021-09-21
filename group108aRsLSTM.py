from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import sys
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

def ccc(x,y):
    '''Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def dispCCC(df):
    # Get CCC
    cccVal = ccc(df.loc[:,df.columns[0]], df.loc[:,df.columns[1]])
    cccVal = np.array2string(cccVal, precision=4)
    print('\t\t'+df.columns[0]+'\t\t'+df.columns[1])
    print(df.columns[0]+'\t\t'+'1.0000'+'\t\t'+cccVal)
    print(df.columns[1]+'\t'+cccVal+'\t\t'+'1.0000')

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

if usingJL:
    jlTag = 'aRs2j'
else:
    jlTag = 'aRs'

# read datasets, first n_steps data will be skipped
# Possible columns: Time,Valence,Arousal,RMS,F0,MFCC1,MFCC2,MFCC3,MFCC4,MFCC5,FileName,voiceTag
if usingJL:
    trainingDataset = pd.read_csv('inputFile/modelInput/allFileCombineP.csv')
    targetOfTrainingDataset = trainingDataset['Arousal'][n_steps:]
    trainingDataset = trainingDataset[['RMS', 'F0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5']]
    print(trainingDataset.head(5))

    testingDataset = pd.read_csv('inputFile/modelInput/jlco0000st.csv')
    targetOfTestingDataset = testingDataset['Arousal'][n_steps:]
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
    # testingyScaled = scaler.fit_transform(np.array(targetOfTestingDataset).reshape(-1, 1))

# split into input and outputs
if usingJL:
    if transformTarget:
        train_X, train_y = train, trainingyScaled[:, 0]
    else:
        train_X, train_y = train, targetOfTrainingDataset

    test_X = test
    test_y = targetOfTestingDataset
else:
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

# Parameters
batch_size = 75
epochs = 70
validation_split = 0.3
# https://machinelearningmastery.com/check-point-deep-learning-models-keras/
usingCheckPoint = False
continueTraining = False

customAdam = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-06,
    amsgrad=False,
    name='Adam',
)

def create_model():
    model = keras.Sequential([
        # no dropout and activation in LSTM if using cuDNN kernel
        layers.LSTM(56, input_shape=(train_X.shape[1], train_X.shape[2])),
        # layers.LSTM(56, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, recurrent_activation='relu'),
        # layers.LSTM(21, recurrent_dropout=0.2,recurrent_activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer=customAdam, loss='mse')
    return model

if not usingCheckPoint:
    # Create a KerasClassifier with best parameters
    model = KerasRegressor(build_fn=create_model, batch_size=batch_size, epochs=epochs, shuffle=False)

    # Calculate the accuracy score for each fold
    kfolds = cross_val_score(model, train_X, train_y, cv=5, scoring='r2')

    # Get the accuracy
    print('The mean accuracy:', kfolds.mean())

# use callbacks
checkpoint = ModelCheckpoint(filepath='outputFile/models/bestWeight.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=14, mode='auto', restore_best_weights=True)

if usingCheckPoint:
    # Create model
    model = create_model()
    # Load weights
    model.load_weights('outputFile/models/bestWeight.hdf5')
    # Compile model (required to make predictions)
    model.compile(optimizer=customAdam, loss='mse')
    print('Created model and loaded weights from file')
    if continueTraining:
        newEpochs = 10
        history = model.fit(train_X, train_y, epochs=newEpochs, batch_size=batch_size, validation_split=validation_split, verbose=2, shuffle=False, callbacks=[early_stop, checkpoint, reduce_lr])
else:
    # Fit network [3, 3, 5, 5, 3433]
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=2, shuffle=False, callbacks=[early_stop, checkpoint, reduce_lr])

# save the model
model.model.save('outputFile/models/'+tTag+jlTag+'Model')
# https://stackoverflow.com/questions/42666046/loading-a-trained-keras-model-and-continue-training

saveOutput = True
# saving output
if saveOutput:
    sys.stdout = open('outputFile/models/'+tTag+jlTag+'Model/0utputLog.txt', "w")

# plot loss history
print(history.history.keys())
plt.ioff()
fig = plt.figure(figsize=[12, 12])
fig.suptitle('Loss Comparison', fontsize=16)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend(loc='upper right')
plt.savefig('outputFile/models/'+tTag+jlTag+'Model/'+'trainLossVsValLoss.png', format='png')
plt.close(fig)

# make a prediction
if transformTarget:
    inv_yPredict = model.predict(test_X)
    # inv transform the predicted value
    yPredict = scaler.inverse_transform(inv_yPredict.reshape(-1, 1))
    yPredict = yPredict[:, 0]
else:
    yPredict = model.predict(test_X)

# actual value
yActual = test_y
# calculate RMSE
rmse = np.sqrt(mean_squared_error(yActual, yPredict))
print('Test RMSE: %.3f' % rmse)

r2_score(yActual, yPredict)

plt.ioff()
pred_test_list = [i for i in yPredict]
submission = pd.DataFrame({'Arousal': yActual, 'Prediction': pred_test_list})
submission.loc[1:, ['Arousal', 'Prediction']].plot(figsize=(36, 24), title='Actual VS Prediction', fontsize=16)
plt.savefig('outputFile/models/'+tTag+jlTag+'Model/'+'actualVsPrediction.png', format='png')
plt.savefig('outputFile/models/'+tTag+jlTag+'Model/'+'actualVsPrediction.svg', format='svg')
plt.close(fig)
submission.to_csv('outputFile/models/'+tTag+jlTag+'Model/'+'submission.csv', index=False)

# print(pred_test_list[1000:1150])

correlation = submission.corr(method='pearson')
print('Pearson Correlation')
print(correlation)
print()
print('Concordance Correlation Coefficient')
dispCCC(submission)

d0 = submission[['Arousal', 'Prediction']]
plt.ioff()
fig = plt.figure(figsize=[24, 24])
fig.suptitle('Actual Prediction Correlation', fontsize=16)
sns.pairplot(d0, kind='scatter')
plt.savefig('outputFile/models/'+tTag+jlTag+'Model/'+'actualPredictionCorrelation.png', format='png')
plt.savefig('outputFile/models/'+tTag+jlTag+'Model/'+'actualPredictionCorrelation.svg', format='svg')
plt.close(fig)

if saveOutput:
    sys.stdout.close()