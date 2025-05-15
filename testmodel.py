import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM


def normalize_data(pd_data):
	divider = (max(pd_data) - min(pd_data))
	return (pd_data - min(pd_data)) / divider
	
	
def denormalize_data(pd_data, min, max):
	multiplier = max - min
	denormalized_data = pd_data * multiplier + min
	return denormalized_data


def create_pack(data, history_size, answer_size):
	X = []
	y = []
	for i in range(len(data) - history_size - answer_size):
		X.append(data[i:(i + history_size)])
		y.append(data[(i + history_size):(i + history_size + answer_size)])
	return np.array(X), np.array(y)


def create_model(history_size, answer_size):
	model = tf.keras.models.Sequential()
	model.add(LSTM(units=50, return_sequences=True, input_shape=(history_size, 1)))
	model.add(LSTM(units=50))
	model.add(Dense(answer_size))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	
	
def train_model(model, data, history_size, answer_size):
	normalized_data = normalize_data(data)
	X_train, Y_train = create_pack(normalized_data, history_size, answer_size)
	X_train = X_train.reshape((X_train.shape[0], history_size, 1))
	model.fit(X_train, Y_train, epochs=10)
	

def create_new_model(data, history_size, answer_size):
	model = create_model(history_size, answer_size)
	train_model(model, data, history_size, answer_size)
	return model
	
data = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
npdata = np.array(data)
model = create_new_model(npdata, 3, 2)

testdata = normalize_data(npdata[:3])
testdata = testdata.reshape((1, 3, 1))
result = model.predict(testdata)
result = denormalize_data(result, 1, 3)
print(result)

