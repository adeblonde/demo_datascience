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

## Second Task

