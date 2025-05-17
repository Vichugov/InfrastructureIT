import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM

# Все массивы (списки) должны иметь тип numpy.array


def normalize_data(pd_data, min, max):
	divider = (max - min)
	return (pd_data - min) / divider
	
	
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
	
	
def train_model(model, data, history_size, answer_size, min, max):
	normalized_data = normalize_data(data, min, max)
	X_train, Y_train = create_pack(normalized_data, history_size, answer_size)
	X_train = X_train.reshape((X_train.shape[0], history_size, 1))
	model.fit(X_train, Y_train, epochs=20)
	

def create_new_model(data, history_size, answer_size, min, max):
	model = create_model(history_size, answer_size)
	train_model(model, data, history_size, answer_size, min, max)
	return model
	
	
def save_model(model, model_name):
	model.save(model_name)	
	
	
def load_model(model_name):
	return tf.keras.models.load_model(model_name)
	
	
def predict_result(model, input_data, min, max):
	normalized_data = normalize_data(input_data, min, max)
	normalized_data = normalized_data.reshape((1, normalized_data.shape[0], 1))
	result = model.predict(normalized_data)
	result = denormalize_data(result, min, max)
	return result
	
