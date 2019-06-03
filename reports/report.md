# Data Analysis Challenge

## First Task

For ease of description, the records with a missing country code are considered from the "null" country.
For each country, we plot the distributions of as many significant parameters as possible. Then, we compare these distributions with the same distributions for the "null" country.
If the "null" country is in truth one of the existing countries, its properties, thus its distributions, should look close to one of the true countries. Nevertheless, there are two other possibilities :

- the "null" country is in fact a __combination__ of the true countries
- the "null" country is a completely different country

For ease of implementation, we focused on the following parameters : 

- distribution of the first city queried by the user (since all records have at least one first city)
- distribution of the unix timestamp of the queries (binned, otherwise the distribution would be quite difficult to analyse graphically)

Our code allows the possibility to add quickly other parameters (joining date, other queried cities, etc), but these first two should be able to give us a good insight.

!['users_by_countries_by_cities'](./figures/users_by_countries_by_cities.png)

*Distribution of the first queried cities - normalized - by countries*

!['users_by_countries_by_timestamps'](./figures/users_by_countries_by_timestamps.png)

*Distribution of the queries' timestamps - normalized - by countries*

It appears that there are no perfectly matching profiles of distributions for the "null" country

![distances_between_countries_by_cities](./figures/distances_between_countries_by_cities.png)

*L2 distances between the 'first cities' profiles of true countries and the 'null' country*

![distances_between_countries_by_timestamps](./figures/distances_between_countries_by_timestamps.png)

*L2 distances between the 'timestamp' profiles of true countries and the 'null' country*

So we tried to consider the 'null' country as a 'good' combination of other countries. A 'good' combination would have the following properties :

- not too many countries involved (i.e. sparse combination)
- positive coefficients (negative coefficients would not have a common sense)

It means Lasso Regression for obtaining the 'first city' and 'timestamps' profile for "null" country as a combination of the others. We could modulate the weights between the parameters (first city, timestamp, or others). We choose an even weighting, and we get the following results for the Lasso Regression :

![lasso_coefs_by_country](./figures/lasso_coefs_by_country.png)

*Coeficients for obtaining the 'null' country as a linear combination of the other countries, with Lasso, and alpha = 2.5*

It appears that the "null" country is probably mainly UK, with elements from DE and US.

## Second Task

We chose to consider the prediction task as a classification one :

- the input is the serie of already requested cities, ordered
- there will be one model for each length of cities' serie (minimum 2, up to 11 in the dataset)
- the expected output is the next city

We need to flatten the column storing the list of queried cities, then we need to split the dataset into as many subsets as possible lengths for the serie of queries cities, for training as much models.
We also need to 'dummify' the 'cities' fields, since it is labelled and qualitative data.
To increase the available number of series for each possible length, when we have a serie of length N, we add all the possible sub-series of length M, M < N, that can be extracted from this serie, to the datasets of series of length M.
Since the number of available series quickly drop to almost zero when their length increases, in practice we will limit our models at series of length 2, 3 and 4.

We will try the following classifiers :

- Logistic Regression
- Linear SVM
- RandomForest

They are all quite quick to learn, and Random Forest do not need a lot of further data preparation or feature building.
Nevertheless, our code is designed in order to allow adding quickly other models.
The target metric will be accuracy, since it is a Multiclass classification.

![Performances_for_cities_serie_length_2](./figures/Performances_for_cities_serie_length_2.png)

*Performances of various Classifier Models in terms of accuracy, for 2-cities long series*

![Performances_for_cities_serie_length_3](./figures/Performances_for_cities_serie_length_3.png)

*Performances of various Classifier Models in terms of accuracy, for 3-cities long series*

![Performances_for_cities_serie_length_4](./figures/Performances_for_cities_serie_length_4.png)

*Performances of various Classifier Models in terms of accuracy, for 4-cities long series*

The observed accuracies seem quite low, but keep in mind that random guess will be quite lower (90 different cities !), and that the number of already available cities is quite low too (<5 cities in each serie)

## Third Task

We try to increase the observed accuracies by adding extra features to the list of already queried cities :

- country
- joining date (more exactly, month of joining date)

We discard the user_id and the other time granularities, for the following reasons :

- either far too much possible values, making dummification intractable (for day of joining date, or for user id)
- or only one possible value (joining year = 2015)

![Extended_Performances_for_cities_serie_length_2](./figures/Extended_Performances_for_cities_serie_length_2.png)

*Performances of various Classifier Models in terms of accuracy, for 2-cities long series*

![Extended_Performances_for_cities_serie_length_3](./figures/Extended_Performances_for_cities_serie_length_3.png)

*Performances of various Classifier Models in terms of accuracy, for 3-cities long series*

![Extended_Performances_for_cities_serie_length_4](./figures/Extended_Performances_for_cities_serie_length_4.png)

*Performances of various Classifier Models in terms of accuracy, for 4-cities long series*

We do not see significant improvement using these extra features

## Fourth Task

We targeted accuracy for the target metric since we are dealing with a Multiclass classification, but we could consider another perspective : if the end user can accept many propositions for the next city, we could suggest a list of cities that are beyond a given threshold of confidence. That will generate extremely high levels of false positives, but it could allow to reduce false negatives (= the suggested city is not the good one). In that case, we should consider a ROC curve.