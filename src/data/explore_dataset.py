# -*- coding: utf-8 -*-
import click
import os
import logging
from pathlib import Path
import json
import pandas as pd
from src.visualization.visualize import bokeh_barchart, bokeh_simple_barchart, bokeh_table
from src.models.predict_model import *
import math
import numpy as np
from sklearn.linear_model import Lasso



def first_city_distribution(logger, df, colors, countries) :

	""" this function computes the distribution of the first city queried among countries """

	""" get list of all 'first' cities """
	cities = df['city_0'].drop_duplicates().tolist()
	logger.info("List of all recorded cities :\n%s" % cities)

	""" occurence of each first city per country """
	data_first_city = dict()

	city_count = df.groupby(['country', 'city_0']).agg({'user_id' : 'count'})
	logger.info(city_count)

	df_cities = pd.DataFrame(cities, columns=['cities'])
	logger.info(df_cities)

	""" we loop through all countries to gather data about the first cities searched """
	for country in countries :
		""" we want a distribution of the users per first city queried in the current country """
		df_first_city = city_count.loc[pd.IndexSlice[country,:]]
		nb_ids_first_city = df_first_city['user_id'].agg('sum')
		logger.info("Nb ids for %s : %s" % (country, nb_ids_first_city))

		df_first_city = df_first_city.merge(df_cities, 'right', left_on='city_0', right_on='cities')
		df_first_city.fillna(0, inplace=True)
		df_first_city = df_first_city.sort_values(by=['cities'])
		logger.info("Number of id by city in %s :\n%s" % (country, df_first_city))

		""" let us normalize the distribution of users """
		df_first_city['user_id'] = df_first_city['user_id'].astype('float').apply(lambda x : x / float(nb_ids_first_city))

		data_first_city['cities'] = sorted(df_first_city['cities'].tolist())
		# print("cities :",data_first_city['cities'])
		data_first_city[country] = df_first_city['user_id'].tolist()

		if country not in colors.keys() :
			colors[country] = 'blue'

	return colors, data_first_city

def timestamp_distribution(logger, df, colors, countries) :

	""" this function computes the distribution of the unix timestamp of the query among countries """

	""" histogram of timestamp """
	timestamp_distrib = df.groupby(['country'])

	""" get list of all timestamps """
	timestamps = df['unix_timestamp'].drop_duplicates().tolist()
	min_timestamp = min(timestamps)
	max_timestamp = max(timestamps)
	nb_timestamps_bins = 20
	timestamp_delta = int((max_timestamp - min_timestamp) / nb_timestamps_bins + 0.5)
	logger.info("max timestamp %s, min timestamp %s, timestamp delta %s" % (max_timestamp, min_timestamp, timestamp_delta))
	timestamps_bins = [timestamp_delta * i + min_timestamp for i in range(nb_timestamps_bins)]
	df_timestamps = pd.DataFrame(timestamps, columns=['timestamps'])

	data_timestamps = dict()

	""" we loop through all countries to gather data about the first cities searched """
	for country in countries :
		""" we want a distribution of the users per binned timestamps queried in the current country """
		df_ts_distrib = df[df['country'] == country]
		nb_ids_timestamp = df_ts_distrib['user_id'].agg('sum')

		df_ts_distrib = df_ts_distrib.merge(df_timestamps, 'right', left_on='unix_timestamp', right_on='timestamps')
		df_ts_distrib.fillna(0, inplace=True)
		df_ts_distrib = df_ts_distrib.sort_values(by=['timestamps'])

		df_ts_distrib['user_id'] = df_ts_distrib['user_id'].astype('float')#.apply(lambda x : x / float(nb_ids_timestamp))

		data_timestamps[country] = df_ts_distrib['unix_timestamp'].tolist()

		data_timestamps[country], data_timestamps['ts_buckets'] = np.histogram(data_timestamps[country], timestamps_bins)
		data_timestamps[country] = data_timestamps[country].tolist()
		data_timestamps['ts_buckets'] = [str(x) for x in data_timestamps['ts_buckets'].tolist()]

		if country not in colors.keys() :
			colors[country] = 'blue'

	return colors, data_timestamps

def get_diff2null(logger, data_dim, countries) :

	""" this function computes, for the chosen feature, the difference of distribution between the 'null' country and all the other countries, using L2 norm """

	diff2null = []
	nnull_countries = []
	for country in countries :
		if country != "null" :
			nnull_countries.append(country)
			diff2null.append(np.linalg.norm(np.array(data_dim[country]) - np.array(data_dim['null'])))

	return diff2null, nnull_countries


def main(raw_data_dir, processed_data_dir, graph_dir, dump_jpg):

	""" Explore dataset and plot analysis """
	logger = logging.getLogger(__name__)
	logger.info('Making cleaned data set from raw data')

	""" defining colors for countries """
	colors = {
		'US' : 'blue',
		'UK' : 'green',
		'FR' : 'yellow',
		'DE' : 'red',
		'IT' : 'purple',
		'ES' : 'orange',
		'null' : 'black'
	}

	""" loading cleaned dataset """
	df = pd.read_csv(os.path.join(processed_data_dir, 'cleaned_dataset.csv'))
	logger.info("All data : %s records" % df.shape[0])

	df_custom = df[df['country'] != ""]
	logger.info("Non null country : %s records" % df_custom.shape[0])
	df_custom = df[df['country'] == ""]
	logger.info("Null country : %s records" % df_custom.shape[0])
	df_custom = '' # avoid spoiling RAM

	df['country'] = df['country'].fillna("null")

	""" get list of all countries """
	countries = sorted(df['country'].drop_duplicates().tolist())
	logger.info("List of all recorded countries :\n%s" % countries)

	""" get the distribution of the first city queried among all countries """
	colors, data_first_city = first_city_distribution(logger, df, colors, countries)

	""" difference betweeen null distribution and other countries' distribution """
	diff2null, nnull_countries = get_diff2null(logger, data_first_city, countries)

	""" plotting the results """
	title_1 = 'Normalized distributions of users by countries and by cities'
	title_2 = 'L2 distance of distribution of users to "null country" records to other countries'

	bokeh_barchart(data_first_city, 'cities', countries, colors, title_1, graph_dir, 'users_by_countries_by_cities', dump_jpg, True)

	bokeh_simple_barchart(nnull_countries, diff2null, title_2, graph_dir, 'distances_between_countries_by_cities', dump_jpg, True)

	""" get the distribution of the timestamp of the query among all countries """
	colors, data_timestamps = timestamp_distribution(logger, df, colors, countries)

	""" difference betweeen null distribution and other countries' distribution """
	diff2null, nnull_countries = get_diff2null(logger, data_timestamps, countries)

	""" plotting the results """
	title_1 = 'Normalized distributions of users by countries and by timestamp'
	title_2 = 'L2 distance of distribution of users to "null country" records to other countries'

	bokeh_barchart(data_timestamps, 'ts_buckets', countries, colors, title_1, graph_dir, 'users_by_countries_by_timestamps', dump_jpg, True)

	bokeh_simple_barchart(nnull_countries, diff2null, title_2, graph_dir, 'distances_between_countries_by_timestamps', dump_jpg, True)

	""" creating aggregated representations of countries using both their first city queried and their timestamps distributions """
	""" here, beta represents the balance between 'first city' and 'timestamps' for the characterization power of a country """
	beta = 0.5
	countries_dist_rep = []
	null_dist_rep = []
	nnull_countries = []
	for country in countries :
		country_rep = [beta * x for x in data_first_city[country]] + [(1 - beta) * x for x in data_timestamps[country]]
		if country == "null" :
			null_dist_rep = np.array(country_rep)
			# null_dist_rep = country_rep
			continue
		else :
			nnull_countries.append(country)
			countries_dist_rep.append(country_rep)

	countries_dist_rep = np.array(countries_dist_rep)
	countries_dist_rep = np.transpose(countries_dist_rep)

	logger.info("Checking input shapes for Lasso : %s and %s" % (null_dist_rep.shape, countries_dist_rep.shape))

	alpha = 2.5
	lasso = Lasso(alpha=alpha)
	lasso.fit(countries_dist_rep, null_dist_rep)

	logger.info("Parameters found for Lasso : %s" % lasso.coef_)

	columns = [('country', 'Country', nnull_countries),
	('coef', 'Lasso coef', lasso.coef_)]
	for i in range(len(nnull_countries)) :
		print("Country : %s, coefficient : %s" % (nnull_countries[i], lasso.coef_[i]))

	bokeh_table(columns, graph_dir, 'lasso_coefs_by_country', dump_jpg, True)

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
	dump_jpg = True

	main(raw_data_dir, processed_data_dir, graph_dir, dump_jpg)