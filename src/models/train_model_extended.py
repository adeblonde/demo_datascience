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

from src.visualization.visualize import bokeh_table

def main(raw_data_dir, processed_data_dir, model_dir, graph_dir, train_ratio, dump_jpg, dump_only):

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
	extra_features = ['joining_date_month', 'country']
	extra_features_values = dict()
	for feature in extra_features :
		feature_values = load_pickle(os.path.join(processed_data_dir, feature + '_data.pkl'))
		extra_features_values[feature] = feature_values


	for i in range(2, nb_max_cities) :
		data_cities = load_pickle(os.path.join(processed_data_dir, 'cities_series_length_' + str(i) + '.pkl'))
		for feature in extra_features :
			data_cities = zip(data_cities, extra_features_values[feature])

			data_cities = [(np.concatenate((x[0], y)), x[1]) for (x, y) in data_cities]

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

		logger.info("Number of training records")
		records_sample = random.sample(range(len(y_cities)), nb_max_records)
		x_cities = x_cities[:nb_max_records]
		y_cities = y_cities[:nb_max_records]
		logger.info("Input sample X:\n%s" % (x_cities[:10]))
		logger.info("Input sample Y:\n%s" % (y_cities[:10]))

		all_performances_train = []
		all_performances_eval = []
		all_models_names = []

		for model_name, model in models.items() :
			""" proceed to training """
			if dump_only == False :
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

				with open(os.path.join(model_dir, 'extended_best_params_' + model_name + '_serie_length_' + str(i) + '.json'), 'w') as f :
					json.dump(search.best_params_, f)

				perfs = {
					'training_score' : search.best_score_,
					'eval_score' : eval_accuracy
				}
				with open(os.path.join(model_dir, 'extended_performances_' + model_name + '_serie_length_' + str(i) + '.json'), 'w') as f :
					json.dump(perfs, f)

				save_pickle(search.best_estimator_, os.path.join(model_dir, 'extended_predictor_%s_cities_series_%s.pkl' % (i, model_name)))

			with open(os.path.join(model_dir, 'extended_performances_' + model_name + '_serie_length_' + str(i) + '.json'), 'r') as f :
							perfs = json.load(f)

			all_performances_train.append(perfs['training_score'])
			all_performances_eval.append(perfs['eval_score'])
			all_models_names.append(model_name)

		""" plot the chart with the performances """
		columns = [('models', 'Models', all_models_names),
		('accuracy_train', 'Acc @ Train', all_performances_train),
		('accuracy_eval', 'Acc @ Eval', all_performances_eval)]
		bokeh_table(columns, graph_dir, 'Extended_Performances_for_cities_serie_length_' + str(i), dump_jpg, True)

if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	project_dir = Path(__file__).resolve().parents[2]

	""" directory containing raw data """
	raw_data_dir = os.path.join(project_dir, 'data', 'raw')

	""" directory containing processed data """
	processed_data_dir = os.path.join(project_dir, 'data', 'processed')

	""" directory containing all the figures generated by the code """
	graph_dir = os.path.join(project_dir, 'reports', 'figures')

	""" directory containing all the trained models and metadata about them """
	model_dir = os.path.join(project_dir, 'models')

	""" do you want to dump jpg figures ? """
	dump_jpg = True

	""" ratio of training records among all available records """
	train_ratio = 0.8

	""" set to True if you don't want to train, just to plot the tables with the performances previously computed """
	# dump_only = True
	dump_only = False

	main(raw_data_dir, processed_data_dir, model_dir, graph_dir, train_ratio, dump_jpg, dump_only)