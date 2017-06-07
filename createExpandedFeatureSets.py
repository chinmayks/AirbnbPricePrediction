"""
createExpandedFeatureSets.py
authors: Akshay Sharma , Chinmay Singh
description: Creates teh expanded feature sets
"""

import pandas as pd

train = listings.sample(frac=0.8, random_state=200)
test = listings.drop(train.index)

train.to_csv(path_or_buf='training_data.csv')
test.to_csv(path_or_buf='testing_data.csv')

neighbourhoods = train['neighbourhoods_cleansed'].unique()

neighbourhoods_unique = []
for item in train['neighbourhoods_cleansed'].unique():
	neighbourhoods_unique.append(item)

neighbourhood_numeric = []
for item in range(len(train['neighbourhoods_cleansed'])):
	for name in neighbourhoods_unique:
		if train['neighbourhoods_cleansed'][item] == name:
			neighbourhood_numeric.append(name)

train['neighbourhood_numeric'] = neighbourhood_numeric

neighbourhood_numeric = []
for item in range(len(test['neighbourhoods_cleansed'])):
	for name in neighbourhoods_unique:
		if test['neighbourhoods_cleansed'][item] == name:
			neighbourhood_numeric.append(name)

test['neighbourhood_numeric'] = neighbourhood_numeric

train = train.fillna(method='ffill')
test= test.fillna(method='ffill')

train.to_csv(path_or_buf='training_data_expanded.csv')
test.to_csv(path_or_buf='testing_data_expanded.csv')