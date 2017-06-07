"""
dataExploration.py
authors: Akshay Sharma , Chinmay Singh
description: Explores the Dataset
"""

import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

BNB_BLUE ='#007A87'
BNB_RED = '#FF5A5F'
BNB_DARY_GREY = '#565A5C'
BNB_LIGHT_GREY = '#CED1CC'

#Exploration of the dataset
listings = pd.read_csv('listings.csv', delimiter=',')
print(listings.columns.values)
pd.options.display.max_columns = 5
y=listings[['price']]
del(listings['price'])
listings = listings.join(y)
print('Number of entries:', listings.shape[0])
print('Number of features:', listings.shape[1]-1)
listings.head(n=3)

def plot_hist(n, titles, ranges):
    fig, ax = plt.subplots(n, figsize = (8, 7.5))
    for i in range(n):
        d, bins, patches = ax[i].hist(ranges[i], 50, normed = 1, color = '#FF5A5F', alpha = 0.85)
        ax[i].set_title(titles[i])
        ax[i].set_xlabel("Daily Price")
        ax[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
prices = listings['price']
plot_hist(3, ['Distribution of Listing Prices: All Data', 'Distribution of Listing Prices: \$0 - \$1000', 'Distribution of Listing Prices: \$0 - \$200'], [prices, prices[prices <= 1000], prices[prices < 250]])

roomTypes = Counter(listings.room_type)
tdf = pd.DataFrame.from_dict(roomTypes, orient = 'index').sort_values(by = 0)
tdf = (tdf.iloc[-10:,:]/listings.shape[0])*100
tdf.sort_index(axis=0, ascending=True,inplace=True)
ax = tdf.plot(kind='bar', figsize=(12, 4), color = '#007A87', alpha = 0.85)      
ax.set_xlabel("Type of Room")
ax.set_ylabel("Percentage of listings")
ax.set_title("Percentage of Listings by Type of Room")
ax.legend_.remove()
plt.show()

print("Private Room Listings: %{0:.2f}".format(tdf[0][0]))
print("Entire home/apt Listings: %{0:.2f}".format(tdf[0][1]))
print("Shared Room Listings: %{0:.2f}".format(tdf[0][2]))

BNB_BLUE ='#007A87'
BNB_RED = '#FF5A5F'
BNB_DARY_GREY = '#565A5C'
BNB_LIGHT_GREY = '#CED1CC'

intervals = [0,100,200,300,1000]
leg_labels=[]

for i in range(0,len(intervals) - 1):
    if i == len(intervals) - 2:
        leg_labels.append('\${}+'.format(intervals[i]))
    else:
        leg_labels.append("\${}-\${}".format(intervals[i], intervals[i+1]))
        
buckets=[]

for i in range(0, len(intervals) - 1):
    buckets.append(listings[(prices > intervals[i]) & (prices < intervals[i+1])])
    
colors = [BNB_LIGHT_GREY, BNB_DARY_GREY, BNB_BLUE, BNB_RED]

alphas= [0.85, 0.85, 0.85, 0.85]
plt.figure(figsize=(15, 15))
for i in range(0, len(buckets)):
    plt.scatter(buckets[i]['longitude'], buckets[i]['latitude'], alpha = alphas[i], c=colors[i], s=25)

plt.title('NYC Airbnb Listings')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend(labels=leg_labels, loc='best')
plt.xlim(-74.2, -73.7)
plt.ylim(40.45, 40.95)
plt.show()