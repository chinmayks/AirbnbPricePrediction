"""
proximityToTimesSquare.py
authors: Akshay Sharma (acs1246), Chinmay Singh (cks9089@rit.edu)
description: Creates a new feature which represents how distant the listing is from Times Square
"""

import pandas as pd
import numpy as np
import scipy as sp
import re
import csv
import nltk
import seaborn as sns
from datetime import datetime
from matplotlib.path import Path
import matplotlib.patches as patches
import sklearn.metrics as Metrics
import matplotlib.pyplot as plt
from collections import Counter

BNB_BLUE = '#007A87'
BNB_RED = '#FF5A5F'
BNB_DARY_GREY = '#565A5C'
BNB_LIGHT_GREY = '#CED1CC'

listings = pd.read_csv('testing_data.csv', delimiter=',', encoding='ISO-8859-1')
caledar = pd.read_csv('calendar.csv',delimiter=',',usecols=range(4) )

# Collect street names from the data
streets = listings['street'].tolist()
streetsClean = []
for i in streets:
    count = str(i).find(',')
    streetsClean.append(str(i)[:count])

listings['streets_clean'] = streets_clean

# List of streets around Times Square
times_streets = ['West 40th Street','West 41st Street','West 42nd Street','West 43rd Street','West 44th Street','West 45th Street',
                 'West 46th Street','West 47th Street','West 48th Street','West 49th Street','West 50th Street']
listings['times'] = listings['streets_clean'].isin(times_streets)

# Check how many are present right around Times Square
times_square = listings[listings['times'] == True]

#Outline of Times Square
verts = [(-73.985628,40.762945),
         (-73.976706,40.759257),
         (-73.753427,40.980884),
         (-73.989837,40.757211),
         (0.,0.)]

#Draw the path
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY]

path =Path(verts,codes)
patch = patches.PathPatch(path, facecolor='none', lw=2)

#Create feature based on the estimated distance of the listing from Times Square
times=[]
for index, row in listings.iterrows():
    if path.contains_point((row['longitude'], row['latitude']), radius=-0.005):
        times.append(1)
    elif path.contains_point((row['longitude'], row['latitude']), radius=-0.010):
        times.append(0.8)
    elif path.contains_point((row['longitude'], row['latitude']), radius=-0.015):
        times.append(0.7)
    elif path.contains_point((row['longitude'], row['latitude']), radius=-0.025):
        times.append(0.55)
    elif path.contains_point((row['longitude'], row['latitude']), radius=-0.05):
        times.append(0.35)
    elif path.contains_point((row['longitude'], row['latitude']), radius=-0.075):
        times.append(0.2)
    elif path.contains_point((row['longitude'], row['latitude']), radius=-0.1):
        times.append(0.1)
    else:
        times.append(0)

listings['times'] = times
exportedData = pd.DataFrame({'id':listings['id'].values, 'times_square': listings['times'].values })

exportedData.to_csv('feature_ts_testing.csv')


features = pd.read_csv('feature_ts_testing.csv')
distance_from_times_square = features['times_square']
listingsSubset['distance_from_times_square'] = distance_from_times_square
listingsSubset.to_csv('feature_data_testing.csv')