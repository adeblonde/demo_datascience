# -*- coding: utf-8 -*-
import click
import os
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from src.common_tools import save_pickle, load_pickle

def subseries(logger, city_list, nb_cities, encoder) :

	""" this function transform a list of cities of length n into as many as possible sets
	of consecutive cities of length nb_cities, with the last one begin separated.
	The output is a tuple :
	([city_0, city_1, ... , city_(nb_cities-1)], city_(nb_cities))
	"""

	try :
		assert(nb_cities > 1)
	except :
		logger.info("Cities series should be > 1")
		raise

	try :
		assert(len(city_list) >= nb_cities)
	except :
		logger.info("List of cities shorter than length of series asked")
		raise

	all_cities_serie = []
	for i in range(len(city_list) - nb_cities + 1) :
		cities_serie = city_list[i:i+nb_cities]
		cities_serie = encoder.transform(cities_serie)
		# cities_serie = [encoder.transform(x) for x in cities_serie]
		all_cities_serie.append(np.concatenate(cities_serie))
		# all_cities_serie.append((cities_serie[:-1], cities_serie[-1]))

	return all_cities_serie

def main(raw_data_dir, processed_data_dir, graph_dir, dump_jpg):

	""" Prepare datasets for next city prediction """

	logger = logging.getLogger(__name__)
	logger.info("Beginning preparing datasets for model training")

	df = pd.read_csv(os.path.join(processed_data_dir, 'cleaned_dataset.csv'))

	""" create series of cities """
	df['cities_serie'] = df['cities'].apply(lambda x : x.split(','))
	df['cities_serie_length'] = df['cities_serie'].apply(lambda x : len(x))
	# df = df.groupby(['cities_serie_length'])

	""" get list of all cities """
	all_cities = []
	nb_max_cities = 0
	for column in df.columns.tolist() :
		if 'city_' in column :
			nb_max_cities += 1
			cities = df[column].drop_duplicates().fillna("null").tolist()
			# print(cities)
			for city in cities :
				if city not in all_cities :
					all_cities.append(city)
	all_cities = sorted(all_cities)
	card_cities = len(all_cities)
	logger.info("List of all %s recorded cities :\n%s" % (card_cities, all_cities))

	""" create Label Binarizer """
	encoder = LabelBinarizer()
	encoder.fit(all_cities)

	""" save parameters to disk """
	cities_parameters = {
		'all_cities' : all_cities,
		'nb_max_cities' : nb_max_cities
	}
	
	with open(os.path.join(processed_data_dir, "cities_parameters.json"), 'w') as f :
		json.dump(cities_parameters, f)

	save_pickle(encoder, os.path.join(processed_data_dir, 'city_encoder.pkl'))

	""" extract cities' series from dataframe """
	cities_series = []
	for i in range(2, nb_max_cities) :
		# temp_serie = df[df['cities_serie_length'] >= i]['cities_serie'].apply(lambda x : len(x))
		temp_df = df[df['cities_serie_length'] >= i]['cities_serie']
		logger.info("Number of cities series with length > %s : %s" % (i, temp_df.shape[0]))
		temp_serie = temp_df.apply(lambda x : subseries(logger, x, i, encoder))
		temp_serie = sum(temp_serie, [])
		logger.info("Number of subseries for cities series length %s : %s" % (i, len(temp_serie)))
		logger.info("Shape of subseries of length %s : %s" %(i, temp_serie[0].shape))

		temp_serie = [(x[:(i-1) * card_cities], np.where(x[(i-1) * card_cities:] == 1)[0].item()) for x in temp_serie]

		""" save prepared cities' serie to disk """
		save_pickle(temp_serie, os.path.join(processed_data_dir, 'cities_series_length_' + str(i) + '.pkl'))

	""" extract extra features """
	extract_extra_feature(logger, df)

def extract_extra_feature(logger, df) :

	""" labelbinarization of extra features ; user id, joining date, country """
	extra_features = ['user_id', 'joining_date_month', 'country']

	for feature in extra_features :
		feature_all_values = df[feature].drop_duplicates().fillna("null").tolist()
		feature_all_values = list(set(feature_all_values))
		if len(feature_all_values) > 50 :
			logger.info("Number of different values for feature %s is %s : too high, drop feature" % (feature, len(feature_all_values)))
			continue
		logger.info("List of all values for feature %s : %s" %(feature, feature_all_values))
		feature_encoder = LabelBinarizer()
		feature_encoder.fit(feature_all_values)
		encoded_features = feature_encoder.transform(df[feature].tolist())
		# encoded_features = feature_encoder.transform(df[feature]).tolist()
		encoded_features = np.array(encoded_features)

		logger.info("Shape of encoded feature %s : %s" % (feature, encoded_features.shape))
		logger.info("Sample of encoded feature %s : %s" % (feature, encoded_features[:3]))

		""" save prepared extra feature to disk """
		save_pickle(encoded_features, os.path.join(processed_data_dir, feature + '_data.pkl'))

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

	""" do you want to dump jpg figures ? """
	dump_jpg = False

	main(raw_data_dir, processed_data_dir, graph_dir, dump_jpg)