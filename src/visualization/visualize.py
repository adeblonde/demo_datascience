import bokeh
from bokeh.models import ColumnDataSource, FactorRange, LabelSet
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.plotting import figure, show, output_file, reset_output
from bokeh.layouts import column, widgetbox
from bokeh.io import export_png
from bokeh.transform import factor_cmap, dodge
from bokeh.palettes import Spectral7, Spectral6
from bokeh.core.properties import value
import os

def bokeh_table(columns, graph_dir, graph_name, dump_jpg, show_html, width=300, height=200):

	"""Dump the (column name, column values) 'columns' dictionary
	as a bokeh table

	Parameters :
		properties (list) : List of columns to plot, with the format [(column_name, column_title, column_values)]
		column_name being a string, column_values being a list
		graph_dir (string) : folder where the output html and/or jpg figure will be dumped
		graph_name (string): name of the figure, the output in 'graph_dir' will be name
		'graph_name.html' and 'graph_name.jpg'
		dum_jpg (boolean) : do not dump the jpg figure if set to False (default is True)
		width (integer) : width of the figure
		height (integer) : height of the figure
	"""

	""" create an output graphics file """
	output_file(os.path.join(graph_dir, graph_name + ".html"))

	dataset = { col_name : col_values for (col_name, _ , col_values) in columns}
	source = ColumnDataSource(dataset)

	table_columns = [TableColumn(field=col_name, title=col_title) for (col_name, col_title , col_values) in columns]

	data_table = DataTable(source=source, columns=table_columns, width=width, height=height)

	p = widgetbox(data_table, sizing_mode='fixed')

	""" show figure in browser """
	if show_html == True :
		show(p)

	""" dump figure as png file """
	if dump_jpg == True :
		export_png(p, filename=os.path.join(graph_dir, graph_name + ".png"))

	reset_output()

def bokeh_simple_barchart(absciss, ordinate, title, graph_dir, graph_name, dump_jpg, show_html, width=1800, height=800) :

	""" Dump simple bokeh barchart with single categorical value """

	""" create an output graphics file """
	output_file(os.path.join(graph_dir, graph_name + ".html"))

	p = figure(x_range = absciss, plot_height=height, plot_width=width, title=title, toolbar_location=None, tools="")

	p.vbar(x=absciss, top=ordinate, width=0.9)

	p.xgrid.grid_line_color = None
	p.y_range.start = 0

	""" show figure in browser """
	if show_html == True :
		show(p)

	""" dump figure as png file """
	if dump_jpg == True :
		export_png(p, filename=os.path.join(graph_dir, graph_name + ".png"))

	reset_output()



def bokeh_barchart(data, absciss, categories, colors, title, graph_dir, graph_name, dump_jpg, show_html, width=1800, height=800):

	""" Dump stacked barcharts as a bokeh chart

	Parameters :
		data (dict) : a dictionary of the form
		data = {
			'absciss' : [absciss values],
			'first_category' : [ordinate values],
			'second_category' : [ordinate_values],
			...
		}
		absciss (string) : the name of the key associated with absciss in the data dict
		categories (list) : a list of string, each string being a category name, in the order of plotting
		colors (dict) : a dict of string, each string being the color associated with the key category
		title (string) : graph title
		graph_dir (string) : folder where the output html and/or jpg figure will be dumped
		graph_name (string): name of the figure, the output in 'graph_dir' will be name
		'graph_name.html' and 'graph_name.jpg'
		dum_jpg (boolean) : do not dump the jpg figure if set to False (default is True)
		width (integer) : width of the figure
		height (integer) : height of the figure
	"""

	# print(data[absciss])
	# print(categories)

	x = [(absciss_value, cat_value) for absciss_value in data[absciss] for cat_value in categories]
	# counts = data[categories[0]]
	# for cat_value in categories :
		# counts = zip(counts, data[cat_value])
		# counts = sum(counts, ())
	# print(counts)
	counts = sum(zip(*[data[cat_value] for cat_value in categories]), ())

	source = ColumnDataSource(data=dict(x=x, counts=counts))

	""" create an output graphics file """
	output_file(os.path.join(graph_dir, graph_name + ".html"))

	p = figure(x_range=FactorRange(*x), plot_height=height, plot_width=width, title=title, toolbar_location=None, tools="")

	customPalette = Spectral6
	customPalette.append('#000000')

	p.vbar(x='x', top='counts', width=0.9, source=source, fill_color=factor_cmap('x', palette=customPalette, factors=categories, start=1, end=2))

	# source = ColumnDataSource(data=data)

	# p = figure(x_range=data[absciss], plot_height=height, plot_width=width, title=title)

	# category_count = 0
	# nb_categories = float(len(categories))
	# print("nb categories : %s" % nb_categories)
	# for category in categories :
	# 	dodge_value = -0.05 * nb_categories + float(category_count) * 0.125
	# 	print("dodge_value : %s" % dodge_value)
	# 	p.vbar(x=dodge(absciss, dodge_value, range=p.x_range), top=category, width=0.1, source=source, color=colors[category], legend=value(category))
	# 	category_count += 1

	p.xgrid.grid_line_color = None
	p.y_range.start = 0
	p.y_range.range_padding = 1
	p.x_range.range_padding = 0.1	
	p.xaxis.major_label_orientation = 1
	p.xgrid.grid_line_color = None
	p.legend.location = "top_left"
	p.legend.orientation = "horizontal"

	""" show figure in browser """
	if show_html == True :
		show(p)

	""" dump figure as png file """
	if dump_jpg == True :
		export_png(p, filename=os.path.join(graph_dir, graph_name + ".png"))

	reset_output()