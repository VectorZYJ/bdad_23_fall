import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import r2_score

import random
random.seed(42)
df = pd.read_csv('aa.csv')


scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_features = scaler.fit_transform(df.drop(['date', 'open_close_diff'], axis=1))
scaler_label = MinMaxScaler(feature_range=(0, 100))
scaled_label = scaler_label.fit_transform(df[['open_close_diff']])
X, y = scaled_features, scaled_label

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

split_percent = 0.80
split = int(split_percent*len(df))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=1, epochs=25)

predictions = model.predict(X_test)

predictions = scaler_label.inverse_transform(predictions)
y_test = scaler_label.inverse_transform(y_test)

print(predictions)
print(y_test)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
r_squared = r2_score(y_test, predictions)
print("R-squared Score:", r_squared)
