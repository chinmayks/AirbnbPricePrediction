"""
sentimentAnalysis.py
authors: Akshay Sharma , Chinmay Singh
description: Creates a new feature which represents how 'good' the listing is perceived to be based on analysis of the listing's title and its reviews
"""

import numpy as np
import pandas as pd
import scipy as sp
import re
import csv
import nltk
import seaborn as sns
from datetime import datetime
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from collections import Counter


BNB_DARK_GRAY = '#565A5C'
BNB_BLUE = '#007A87'
BNB_LIGHT_GRAY = '#CED1CC'
BNB_RED = '#FF5A5F'

# Read in datasets
listings = pd.read_csv('listings_clean.csv', delimiter=',', encoding='ISO-8859-1')
reviews = pd.read_csv('reviews.csv', delimiter=',')

y = listings['price']

# Append price at the end of the table
del listings['price']
x = listings
              
listings = listings.join(y)

titles = []
for item in listings['name'].values:
    if len(item) > 0:
        titles.append(item)
        
x['name'] = titles
        
reviews = reviews.dropna(axis=0)

# Concatenate title summary and listing review in one column
emptyConcat = []
for item in x['id'].unique():
     if item in reviews['listing_id'].unique():
             if len(x['name'].loc[x['id']==item].values) > 0 and len(reviews['comments'].loc[reviews['listing_id'] == item].values) > 0:
                     emptyConcat.append(str(x['name'].loc[x['id'] == item].values[-1]) + ' ' + str(reviews['comments'].loc[reviews['listing_id'] == item].values[:-1]))
     else:
             if len(x['name'].loc[x['id'] == item].values) > 0:
                     emptyConcat.append(str(x['name'].loc[x['id'] == item].values[-1]))
     
x['neighborhood_names']=neighborhoodWords
 
nbs = x['neighborhood_names'].unique().tolist()

# Stem the words in a listing and add it to the 'bag of words' of that listing

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

bags = empty_concat
bagOfWords = []

# Convert all characters to the English language
for review in bags:
    # Replace any characters that are not English
    replaced = re.sub("[^a-zA-Z]", " ", review)

    lower_case = replaced.lower()   
    words = lower_case.split()

    # Add to the list and exclude stop-words and words specific to neighbourhoods
    words = [stemmer.stem(w) for w in words if ((not w in stops) & (not w in nbs))]
    listing_words = ' '.join(words)     
    bagOfWords.append(listing_words)

bag = pd.DataFrame(x['id'])
bag['bag_of_words'] = bagOfWords

#Create a file
bag.to_csv('bag_of_words.csv')

# Sentiment prediction using:
# http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar

negativeWords = pd.read_csv('negative-words.txt', encoding='latin1', skiprows = 34)
positiveWords = pd.read_csv('positive-words.txt', encoding='latin1', skiprows = 34)

pos_lib_full = positiveWords.iloc[:, 0].tolist()
neg_lib_full = negativeWords.iloc[:, 0].tolist()
new = []

for each in positiveWords.iloc[:, 0].tolist():
    word = each.encode('ascii', 'ignore')
    neg_lib_full.append(word)

for each in negativeWords.iloc[:, 0].tolist():
    word = each.encode('ascii', 'ignore')
    new.append(word)

#Create corpuses to make the process quicker
pos_lib_stems = [stemmer.stem(str(w)) for w in pos_lib_full]
pos_lib = set(pos_lib_full + pos_lib_stems)

neg_lib_full = new
neg_lib_stems = [stemmer.stem(str(w)) for w in neg_lib_full]
neg_lib = set(neg_lib_full + neg_lib_stems)

# Estimate the overall sentiment about a listing from the review
#using the appearance of +ve and -ve words stems and their relative polarities
def predictPolarity(texts):
    polarities = []
    
    for i in range(0, len(texts)):
        opinion = texts[i].split()
        posCount, negCount = 0.0, 0.0
        polarity = 0.5
        for word in opinion:
            if word in pos_lib:
                posCount += 1.0
            elif word in neg_lib:
                negCount += 1.0
        
        if (posCount == 0.0) & (negCount == 0.0):
            pass
        else:
            polarity = posCount/(posCount + negCount)
        polarities.append(polarity)
        
    return polarities

fullBag = bag['bag_of_words'].tolist()
manualPolarities = predictPolarity(fullBag)

#Make use of both NLTK and TextBlob in order to predict sentiment of the bags of words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

nltk_polarities = []

for each in fullBag:
    each = " ".join(each)
    ss = sid.polarity_scores(each)
    nltk_polarities.append(ss.values())
    
from textblob import TextBlob

textBlobPolarities = []

for i, each in enumerate(fullBag):
    blob = TextBlob(each)
    textBlobPolarities.append(blob.sentiment.polarity)

onlySentiments = pd.DataFrame({'id':listings['id'].values, 'polarities': textBlobPolarities})
onlySentiments = onlySentiments[['id', 'polarities']]
onlySentiments.to_csv('sentimentFeature.csv')

listingsSubset = pd.DataFrame({'id':listings['id'].values, 'polarities':textBlobPolarities, 'property_type':listings['property_type'].values, 'room_type':listings['room_type'].values, 'bed_type':listings['bed_type'].values, 'review_scores_rating':listings['review_scores_rating'].values})
listingsSubset.to_csv('features.csv')