from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# prepare data for lstms
from sklearn.preprocessing import MinMaxScaler

# convert series to supervised learning


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


# specify the number of lag hours
n_steps = 24
n_features = 7

dataset = pd.read_csv('../inputFile/modelInput/jlco0000st.csv')
y_train_dataset = dataset['Arousal'][n_steps:]
dataset = dataset.drop(columns=['Time', 'FileName', 'Valence', 'Arousal'])

test_dataset = pd.read_csv(
    '../inputFile/modelInput/sentences/allFileCombineSentenceP.csv')
y_test_dataset = test_dataset['Arousal'][n_steps:]
test_dataset = test_dataset.drop(columns=['Time', 'Valence', 'Arousal'])

# load dataset
values = dataset.values
test_values = test_dataset.values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
test_scaled = scaler.fit_transform(test_values)
# frame as supervised learning
reframed = series_to_supervised(scaled, n_steps, 1)
test_reframed = series_to_supervised(test_scaled, n_steps, 1)

# split into train and test sets
values = reframed.values
test_values = test_reframed.values
# train = values
# test = test_values
n_train_hours = 6030
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
n_obs = (n_steps + 1) * n_features
# train_X, train_y = train[:, :n_obs], y_train_dataset
# test_X, test_y = test[:, :n_obs], y_test_dataset
train_X, train_y = train[:, :n_obs], y_train_dataset[0:n_train_hours]
test_X, test_y = test[:, :n_obs], y_train_dataset[n_train_hours:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_steps + 1, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps + 1, n_features))


def create_model():
    model = keras.Sequential([
        layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])),
        layers.Dense(1),
    ])

    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=0.0001), loss='mean_squared_error')

    return model


# Create a KerasClassifier with best parameters
model_KR = KerasRegressor(build_fn=create_model, batch_size=8, epochs=1000)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model_KR, train_X, train_y, cv=7, scoring='r2')

# get the accuracy
print(kfolds.mean())
print('The mean accuracy:', kfolds.mean())

# use callbacks

checkpoint = ModelCheckpoint(
    "", monitor="val_loss", verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                           patience=5, mode='auto', restore_best_weights=True)

# fit network
history = model_KR.fit(train_X, train_y, epochs=1000, batch_size=8, validation_split=0.2,
                       verbose=2, shuffle=False, callbacks=[early_stop, checkpoint, reduce_lr])
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

# make a prediction
yhat = model_KR.predict(test_X)
# invert scaling for forecast
inv_yhat = yhat
#inv_yhat = inv_yhat[:,0]
# invert scaling for actual
inv_y = y_train_dataset[n_train_hours:]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

r2 = r2_score(inv_y, inv_yhat)
print('Test r2_score: %.3f' % r2)
pred_test_list = [i for i in inv_yhat]
submission = pd.DataFrame({'Arousal': inv_y, 'Prediction': pred_test_list})

submission.loc[1:, ['Arousal', 'Prediction']].plot()
submission.to_csv('submission.csv', index=False)
correlation = submission.corr(method='pearson')
print(correlation)
d0 = submission[['Arousal', 'Prediction']]
sns.pairplot(d0, kind="scatter")
plt.show()
