from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor

import pickle
import time
import pandas as pd
import numpy as np


def train(csv_filename, pred_column, output_prefix):
	algos = list_algos()

	prediction_label = pred_column

	df = pd.read_csv(csv_filename)

	X = np.array(df.drop([prediction_label], 1))
	y = np.array(df[prediction_label])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30)

	scalar = MinMaxScaler().fit(X_train)
	X_train = scalar.transform(X_train)
	X_test = scalar.transform(X_test)
	save(scalar, '{}-TestdecisionTree-{}.scalar'.format(output_prefix, time.time()))

	dummy = DummyRegressor(strategy='mean')
	dummy.fit(X_train, y_train)
	dummy_score = dummy.score(X_test, y_test)
	print('Dummy {}'.format(dummy_score))

	for algo in algos:
		model = algo()
		model.fit(X_train, y_train)

		score = model.score(X_test, y_test)
		print('{} {}'.format(algo, score))
		save(model, '{}-TestdecisionTree-{}.model'.format(output_prefix, time.time()))


def save(obj, filename):
	pickle.dump(obj, open(filename, 'wb'))


def list_algos():
	# return [Lasso, ElasticNet, Ridge, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor]
	return [DecisionTreeRegressor]


if __name__ == '__main__':
	train(
		'india/india_combined.csv',
		'sound_vel',
		'india')
