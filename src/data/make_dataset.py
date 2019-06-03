# -*- coding: utf-8 -*-
import click
import os
import logging
from pathlib import Path
import json
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime

def main(raw_data_dir, processed_data_dir, graph_dir, dump_jpg):
	
	""" Runs data processing scripts to turn raw data from (../raw) into
		cleaned data ready to be analyzed (saved in ../processed).
	"""

	logger = logging.getLogger(__name__)
	logger.info('Making cleaned data set from raw data')

	""" loading raw data source and begin cleaning """
	raw_data = ''
	with open(os.path.join(raw_data_dir, 'city_search.json'), 'r') as fp :
		raw_data = fp.read()
		df = pd.read_json(raw_data, orient='records')

	logger.info("Raw columns' names : %s" % df.columns.tolist())
	logger.info("Raw Data insights : \n%s" % df[:3])

	fields = ['cities', 'session_id', 'unix_timestamp', 'user']

	""" Some fields are lists of length 1 : we flatten them """
	logger.info("Unboxing fields")
	df = field_unboxing(logger, df, fields)
	logger.info("Data after unboxing")
	logger.info(df[:5])

	final_fields = ['cities', 'session_id', 'unix_timestamp', 'country', 'joining_date', 'user_id']

	""" the session_id field is a nested json """
	logger.info("Flattening of json field (session_id)")
	df = user_flattening(logger, df, final_fields)
	logger.info("Data after flattening :\n%s" % df[:3])

	""" we extract time elements from unix timestamp and joining date """
	logger.info("Extraction of time elements from timestamp and joining date")
	df = time_extract(logger, df, 'unix_timestamp', 'timestamp')
	df = time_extract(logger, df, 'joining_date', 'date')
	logger.info("Data after extraction of time elements :\n%s" % df[:3])

	""" we split the 'cities' column """
	logger.info("Splitting cities column into multiple columns, one per city")
	df['cities'] = df['cities'].apply(lambda x : x.replace(', ', ','))
	df_split = pd.DataFrame(df['cities'].str.split(',', expand=True).values.tolist())
	nb_max_cities = df_split.shape[1]
	cities_col_names = [0] * nb_max_cities
	for i in range(nb_max_cities) :
		cities_col_names[i] = 'city_' + str(i)
	df_split.columns = cities_col_names
	df = df.merge(df_split, how='inner', left_index=True, right_index=True)
	logger.info("Data after splitting cities column :\n%s" % df[:3])

	""" checking the process """
	logger.info("Final columns' names : %s" % df.columns.tolist())

	""" writing cleaned dataset to disk """
	logger.info("Dumping cleaned dataset")
	logger.info("All data : %s records" % df.shape[0])
	df.to_csv(os.path.join(processed_data_dir, 'cleaned_dataset.csv'))

def field_unboxing(logger, df, fields) :

	""" create new columns extracting unique element from list inside record """

	df['errors_unboxed_total'] = 0
	for field in fields :
		df['unboxed_' + field] = df[field].apply(lambda x : x[0])
		df['errors_unboxed_' + field] = df[field].apply(lambda x : len(x) - 1)
		df['errors_unboxed_total'] += df['errors_unboxed_' + field]

	""" how many unboxing errors ? """
	logger.info("Total unboxing errors : %s" % df[df['errors_unboxed_total'] > 0].shape[0])
	df = df[df['errors_unboxed_total'] == 0]

	unboxed_fields = ['unboxed_' + field for field in fields]

	df = df[unboxed_fields]
	logger.info("Unboxed columns' names : %s" % df.columns.tolist())
	
	for field in fields :
		df = df.rename(columns={'unboxed_' + field : field})

	return df

def user_flattening(logger, df, final_fields) :

	""" flatten the 'user' fields from a json structure to several columns """

	""" flattening user field """
	df['user'] = df['user'].apply(pd.Series)
	df_user_flat = json_normalize(df['user'])
	logger.info("Flattened columns' names : %s" % df_user_flat.columns.tolist())

	df = df.merge(df_user_flat, how='inner', left_index=True, right_index=True)

	logger.info("Columns' names after flattening: %s" % df.columns.tolist())

	df = df[final_fields]

	# logger.info(df[:5])

	return df

def time_extract(logger, df, column_name, time_mode) :

	""" this function takes a dataframe df as input, and creates the columns :
	column_name_year
	column_name_month
	column_name_day
	column_name_hour
	column_name_minute
	column_name_second
	from the 'column_name' column
	"""

	time_keys = {
		'year' : 'Y',
		'month' : 'm',
		'day' : 'd',
		'hour' : 'H',
		'minute' : 'M',
		'second' : 'S'
	}

	date_keys = {
		'year' : [0, 4],
		'month' : [5, 7],
		'day' : [8, 10]
	}

	if time_mode == 'timestamp' :
		for time_granul, time_key in time_keys.items() :
			df[column_name + '_' + time_granul] = df[column_name].apply(lambda x : datetime.utcfromtimestamp(x).strftime('%' + time_key))
	if time_mode == 'date' :
		for time_granul, time_key in date_keys.items() :
			df[column_name + '_' + time_granul] = df[column_name].apply(lambda x : x[date_keys[time_granul][0]: date_keys[time_granul][1]])
	return df

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
