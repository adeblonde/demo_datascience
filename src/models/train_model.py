import json
from src.common_tools import load_pickle, save_pickle
import os
import logging
from pathlib import Path
import numpy as np
import random

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(raw_data_dir, processed_data_dir, model_dir, graph_dir, train_ratio, dump_jpg):

	""" This function loads pickled prepared data from the prepared_folder
	and creates train models from them 
	"""
	logger = logging.getLogger(__name__)

	""" load cities' parameters """
	with open(os.path.join(processed_data_dir, 'cities_parameters.json'), 'r') as f :
		cities_parameters = json.load(f)

	all_cities = cities_parameters['all_cities']
	nb_max_cities = cities_parameters['nb_max_cities']

	""" create preprocessing Pipeline """
	prep_pipeline = Pipeline([
		('std_scaler', StandardScaler())
	])

	""" models list """
	models = {
		'LogReg' : {
			'model' : LogisticRegression(),
			'param_grid' : {
				'LogReg__C' : [0.1, 1, 10, 100]
			}
		},
		'LinearSVC' : {
			'model' : SVC(kernel='linear', probability=True),
			'param_grid' : {
				'LinearSVC__C' : [0.1, 1, 10, 100]
			}
		},
		'RandomForest' : {
			'model' : RandomForestClassifier(),
			'param_grid' : {
				'RandomForest__n_estimators' : [200, 500],
				'RandomForest__max_features': ['auto', 'sqrt', 'log2']
			}
		}
	}

	""" cross-validation number of folds """
	nb_folds = 5

	""" limiter for debug """
	nb_max_records = 500

	""" load data """
	for i in range(2, nb_max_cities) :
		# x_cities, y_cities = load_pickle(os.path.join(processed_data_dir, 'cities_series_length_' + str(i) + '.pkl'))
		data_cities = load_pickle(os.path.join(processed_data_dir, 'cities_series_length_' + str(i) + '.pkl'))

		""" split into modelization/train dataset and independant evaluation dataset """
		data_cities_model, data_cities_eval = train_test_split(data_cities, train_size=train_ratio, random_state=42)
		data_cities_model = [[*x] for x in zip(*data_cities_model)]
		data_cities_model = [np.array(x) for x in data_cities_model]

		x_cities = data_cities_model[0]
		y_cities = data_cities_model[1]

		data_cities_eval = [[*x] for x in zip(*data_cities_eval)]
		data_cities_eval = [np.array(x) for x in data_cities_eval]

		x_cities_eval = data_cities_eval[0]
		y_cities_eval = data_cities_eval[1]

		""" get records from cities with sufficient total number of records only """
		cities_unique, cities_counts = np.unique(y_cities, return_counts=True)

		ok_cities = cities_unique[np.where(cities_counts >= nb_folds)]

		ok_records = [index for index, city in enumerate(y_cities) if city in ok_cities]

		x_cities = x_cities[ok_records]
		y_cities = y_cities[ok_records]

		# logger.info("Cities with sufficient number of records :\n%s" % ok_cities)
		# logger.info("Records from cities with sufficient number of records :\n%s" % ok_records)

		# y_count = [(x,y_cities_train.count(x)) for x in set(y_cities_train)]
		# y_indices = dict((x, [i for i, e in enumerate(y_cities_train) if e == x]) for x in set(y_cities_train))

		# ok_indices = []

		# for city_key, city_count in y_count :
		# 	if city_count >= nb_folds :
		# 		ok_indices += (y_indices[city_key])

		# logger.info("Indices of cities with sufficient number of records :\n%s" % ok_indices)
		# logger.info("Counts of cities with sufficient number of records :\n%s" % y_count)
		# logger.info("Indices of cities with sufficient number of records :\n%s" % ok_indices)


		logger.info("Number of training records")
		records_sample = random.sample(range(len(y_cities)), nb_max_records)
		x_cities = x_cities[:nb_max_records]
		y_cities = y_cities[:nb_max_records]
		# x_cities = prep_pipeline.fit_transform(x_cities)
		# y_cities = prep_pipeline.fit_transform(y_cities)
		logger.info("Input sample X:\n%s" % (x_cities[:10]))
		logger.info("Input sample Y:\n%s" % (y_cities[:10]))
		for model_name, model in models.items() :
			cities_cf = Pipeline([
				# ('std_scaler', StandardScaler()),
				(model_name, model['model'])
			])
			if 'param_grid' in model.keys() :
				search = GridSearchCV(
					cities_cf,
					model['param_grid'],
					iid=False,
					cv=5,
					scoring='accuracy'
				)
			else :
				search = GridSearchCV(
					cities_cf,
					[],
					iid=False,
					cv=5,
					scoring='accuracy'
				)

			logger.info("Grid searching : model %s, cities serie length %s" % (model_name, i))
			search.fit(x_cities, y_cities)
			logger.info("Best training score : %s" % search.best_score_)

			eval_predicted = search.predict(x_cities_eval)
			eval_accuracy = accuracy_score(y_cities_eval, eval_predicted)

			logger.info("Evaluation score : %s" % eval_accuracy)

			with open(os.path.join(model_dir, 'best_params_' + model_name + '_serie_length_' + str(i) + '.json'), 'w') as f :
				json.dump(search.best_params_, f)

			perfs = {
				'training_score' : search.best_score_,
				'eval_score' : eval_accuracy
			}
			with open(os.path.join(model_dir, 'performances_' + model_name + '_serie_length_' + str(i) + '.json'), 'w') as f :
				json.dump(perfs, f)

			save_pickle(search.best_estimator_, os.path.join(model_dir, 'predictor_%s_cities_series_%s.pkl' % (i, model_name)))

		# cv = StratifiedKFold(n_splits=5)
		# train_set, eval_set = cv.split(x_cities, y_cities)

if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	# load_dotenv(find_dotenv())

	""" directory containing raw data """
	raw_data_dir = os.path.join(project_dir, 'data', 'raw')

	""" directory containing processed data """
	processed_data_dir = os.path.join(project_dir, 'data', 'processed')

	""" directory containing all the figures generated by the code """
	graph_dir = os.path.join(project_dir, 'reports', 'figures')

	""" directory containing all the trained models and metadata about them """
	model_dir = os.path.join(project_dir, 'models')

	""" do you want to dump jpg figures ? """
	dump_jpg = False

	""" ratio of training records among all available records """
	train_ratio = 0.8

	main(raw_data_dir, processed_data_dir, model_dir, graph_dir, train_ratio, dump_jpg)