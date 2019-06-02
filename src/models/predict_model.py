import json
from src.common_tools import load_pickle, save_pickle
import os
import logging
from pathlib import Path
import numpy as np

def main(processed_data_dir, model_dir, inpt, model_name) :

	""" this function takes as inpt a string of the format
	CITY_1, CITY_2, CITY_3, ..., CITY_N
	and print a list of probabilities for what the next city will be, with the format :
	- most probable next city name, associated probability
	- second most probable next city name, associated probability
	- third most probable ...
	"""

	logger = logging.getLogger(__name__)

	""" load cities' parameters """
	with open(os.path.join(processed_data_dir, 'cities_parameters.json'), 'r') as f :
		cities_parameters = json.load(f)

	all_cities = cities_parameters['all_cities']
	nb_max_cities = cities_parameters['nb_max_cities']

	""" load cities' encoder """
	encoder = load_pickle(os.path.join(processed_data_dir, 'city_encoder.pkl'))

	cities_serie = inpt.replace(', ', ',')
	cities_serie = cities_serie.split(',')
	serie_size = len(cities_serie) + 1
	named_cities_serie = cities_serie
	cities_serie = encoder.transform(cities_serie)
	cities_serie = np.array([x for city_vec in cities_serie for x in city_vec ])

	model_path = os.path.join(model_dir, 'predictor_%s_cities_series_%s.pkl' % (serie_size, model_name))

	while os.path.isfile(model_path) is False :
		serie_size += -1
		model_path = os.path.join(model_dir, 'predictor_%s_cities_series_%s.pkl' % (serie_size, model_name))
		if serie_size < 2 :
			logger.info('No model available for this length of cities serie')
			return
	
	logger.info("Prediction %s-th element of serie %s" % (serie_size, named_cities_serie))

	estimator = load_pickle(model_path)

	predicted_probas = estimator.predict_proba([cities_serie])
	logger.info(predicted_probas[0])

	cities_proba = []
	for i in range(len(predicted_probas[0])) :
		cities_proba.append((all_cities[i], predicted_probas[0][i]))

	# logger.info("Raw prediction :\n%s" % cities_proba)

	cities_proba = sorted(cities_proba, key=lambda x : x[1], reverse=True)

	logger.info("Raw prediction :\n%s" % cities_proba)

	print("Predicted next city (in decreasing order of confidence) :")
	for city_proba in cities_proba :
		print("%s : %s" % (city_proba[0], city_proba[1]))

if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	# load_dotenv(find_dotenv())

	""" directory containing processed data """
	processed_data_dir = os.path.join(project_dir, 'data', 'processed')

	""" directory containing all the trained models and metadata about them """
	model_dir = os.path.join(project_dir, 'models')

	""" choose model """
	model_name = 'RandomForest'

	""" test inpt """
	inpt = "San Antonio TX,Corpus Christi TX,Arlington TX"

	main(processed_data_dir, model_dir, inpt, model_name)