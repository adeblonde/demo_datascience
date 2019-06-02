# -*- coding: utf-8 -*-
import click
import os
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import json
import pandas as pd
from src.visualization.visualize import bokeh_barchart, bokeh_simple_barchart
from src.models.predict_model import *
import math
import numpy as np

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(raw_data_dir, processed_data_dir, graph_dir, dump_jpg):

	""" Explore dataset and plot analysis """
	logger = logging.getLogger(__name__)
	logger.info('Making cleaned data set from raw data')

	# df_nnull = pd.read_csv(os.path.join(processed_data_dir, 'non_null_country_dataset.csv'))
	df_nnull = pd.read_csv(os.path.join(processed_data_dir, 'cleaned_dataset.csv'))
	df_null = pd.read_csv(os.path.join(processed_data_dir, 'null_country_dataset.csv'))

	df_nnull['country'] = df_nnull['country'].fillna("null")

	""" get list of all countries """
	countries = sorted(df_nnull['country'].drop_duplicates().tolist())
	logger.info("List of all recorded countries :\n%s" % countries)

	""" get list of all 'first' cities """
	cities = df_nnull['city_0'].drop_duplicates().tolist()
	logger.info("List of all recorded cities :\n%s" % cities)

	""" occurence of each first city per country """
	data_first_city = dict()

	city_count = df_nnull.groupby(['country', 'city_0']).agg({'user_id' : 'count'})
	logger.info(city_count)

	df_cities = pd.DataFrame(cities, columns=['cities'])
	logger.info(df_cities)

	# colors = dict()
	colors = {
		'US' : 'blue',
		'UK' : 'green',
		'FR' : 'yellow',
		'DE' : 'red',
		'IT' : 'purple',
		'ES' : 'orange',
		'null' : 'black'
	}

	for country in countries :
		df_first_city = city_count.loc[pd.IndexSlice[country,:]]
		nb_ids_first_city = df_first_city['user_id'].agg('sum')
		logger.info("Nb ids for %s : %s" % (country, nb_ids_first_city))

		df_first_city = df_first_city.merge(df_cities, 'right', left_on='city_0', right_on='cities')
		df_first_city.fillna(0, inplace=True)
		df_first_city = df_first_city.sort_values(by=['cities'])
		logger.info("Number of id by city in %s :\n%s" % (country, df_first_city))

		df_first_city['user_id'] = df_first_city['user_id'].astype('float').apply(lambda x : x / float(nb_ids_first_city))

		data_first_city['cities'] = sorted(df_first_city['cities'].tolist())
		print("cities :",data_first_city['cities'])
		data_first_city[country] = df_first_city['user_id'].tolist()

		if country not in colors.keys() :
			colors[country] = 'blue'

	diff2null = []
	nnull_countries = []
	for country in countries :
		if country != "null" :
			nnull_countries.append(country)
			diff2null.append(np.linalg.norm(np.array(data_first_city[country]) - np.array(data_first_city['null'])))

	title_1 = 'Normalized distributions of users by countries and by cities'
	title_2 = 'L2 distance of distribution of users to "null country" records to other countries'

	bokeh_barchart(data_first_city, 'cities', countries, colors, title_1, graph_dir, 'users_by_countries_by_cities', dump_jpg, True)

	bokeh_simple_barchart(nnull_countries, diff2null, title_2, graph_dir, 'users_by_countries_by_cities', dump_jpg, True)


	""" histogram of timestamp """
	timestamp_distrib = df_nnull.groupby(['country'])
	# timestamp_distrib = df_nnull.groupby(['country', 'unix_timestamp']).agg({'user_id' : 'count'})

	""" get list of all timestamps """
	timestamps = df_nnull['unix_timestamp'].drop_duplicates().tolist()
	min_timestamp = min(timestamps)
	max_timestamp = max(timestamps)
	nb_timestamps_bins = 20
	timestamp_delta = int((max_timestamp - min_timestamp) / nb_timestamps_bins + 0.5)
	logger.info("max timestamp %s, min timestamp %s, timestamp delta %s" % (max_timestamp, min_timestamp, timestamp_delta))
	timestamps_bins = [timestamp_delta * i + min_timestamp for i in range(nb_timestamps_bins)]
	df_timestamps = pd.DataFrame(timestamps, columns=['timestamps'])

	data_timestamps = dict()

	for country in countries :
		# df_ts_distrib = timestamp_distrib.loc[pd.IndexSlice[country,:]]
		df_ts_distrib = df_nnull[df_nnull['country'] == country]
		nb_ids_timestamp = df_ts_distrib['user_id'].agg('sum')

		df_ts_distrib = df_ts_distrib.merge(df_timestamps, 'right', left_on='unix_timestamp', right_on='timestamps')
		df_ts_distrib.fillna(0, inplace=True)
		df_ts_distrib = df_ts_distrib.sort_values(by=['timestamps'])

		df_ts_distrib['user_id'] = df_ts_distrib['user_id'].astype('float')#.apply(lambda x : x / float(nb_ids_timestamp))

		# data_timestamps['timestamps'] = sorted(df_ts_distrib['timestamps'].tolist())
		data_timestamps[country] = df_ts_distrib['unix_timestamp'].tolist()
		# print(data_timestamps[country][:100])
		# print(timestamps_bins)

		data_timestamps[country], data_timestamps['ts_buckets'] = np.histogram(data_timestamps[country], timestamps_bins)
		print(data_timestamps[country])
		data_timestamps[country] = data_timestamps[country].tolist()
		data_timestamps['ts_buckets'] = [str(x) for x in data_timestamps['ts_buckets'].tolist()]


		if country not in colors.keys() :
			colors[country] = 'blue'

	diff2null = []
	nnull_countries = []
	for country in countries :
		if country != "null" :
			nnull_countries.append(country)
			diff2null.append(np.linalg.norm(np.array(data_timestamps[country]) - np.array(data_timestamps['null'])))

	title_1 = 'Normalized distributions of users by countries and by timestamp'
	title_2 = 'L2 distance of distribution of users to "null country" records to other countries'

	# print(data_timestamps)

	bokeh_barchart(data_timestamps, 'ts_buckets', countries, colors, title_1, graph_dir, 'users_by_countries_by_timestamps', dump_jpg, True)

	bokeh_simple_barchart(nnull_countries, diff2null, title_2, graph_dir, 'users_by_countries_by_timestamps', dump_jpg, True)



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

	""" do you want to dump jpg figures ? """
	dump_jpg = False

	main(raw_data_dir, processed_data_dir, graph_dir, dump_jpg)