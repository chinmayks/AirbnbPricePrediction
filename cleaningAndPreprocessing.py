"""
cleaningAndPreprocessing.py
authors: Akshay Sharma , Chinmay Singh
description: Cleans and pre-processes the given dataset as much as possible to ensure that we have clean data for feature creation, extraction and modelling
"""

# Read in the dataset
listings = pd.read_csv('listings.csv', delimiter=',')

# Remove price to ensure it remains stored away from any changes
y = listings['price']
del listings['price']

# Store the number of rows and columns present
entries = listings.shape[0]
features = listings.shape[1]

# These were found to either have too few actual values or be unnecessary in the long term
badFeatures = ['scrape_id', 'last_scraped', 
'experiences_offered', 'thumbnail_url', 'medium_url', 'xl_picture_url',
 'host_id', 'host_url', 'host_name', 'host_since', 'host_location',
 'host_thumbnail_url', 'neighbourhood', 'state', 'market', 'country_code',
 'country', 'weekly_price', 'monthly_price', 'first_review', 'last_review'
 ,'square_feet', 'license', 'host_acceptance_rate', 'jurisdiction_names',
 'has_availability']

listings.drop(badFeatures, axis=1, inplace=True)
features = listings.shape[1]

# Check the percentage of a column that is empty
def emptinessPercent(df):
    bools = df.isnull().tolist()
    percentEmpty = float(bools.count(True)) / float(len(bools))
    return  percentEmpty, float(bools.count(True))

emptiness = []
missingColumns = []

# Get emptiness for each feature present
for ctr in range(0, listings.shape[1]):
    p, n = emptinessPercent(listings.iloc[:,ctr])
    if n>0:
        missingColumns.append(listings.columns.values[ctr])
    emptiness.append(round((p),2))

# Plot a graph to show the emptiness percentage of each feature
empty_dict = dict(zip(listings.columns.values.tolist(), emptiness))
empty = pd.DataFrame.from_dict(empty_dict, orient = 'index').sort_values(by=0)
ax = empty.plot(kind = 'bar', color='#E35A5C', figsize = (16, 5))
ax.set_xlabel('Predictor')
ax.set_ylabel('Percent Empty / NaN')
ax.set_title('Feature Emptiness')
ax.legend_.remove()
# plt.show()

features = listings.shape[1]

# Identify columns that are numeric but not of that type, and convert columns into floats
to_float = ['id', 'latitude', 'longitude', 'accommodates',
'bathrooms', 'bedrooms', 'beds', 'guests_included',
'extra_people', 'minimum_nights', 'maximum_nights',
'availability_30', 'availability_60', 'availability_90',
'availability_365', 'number_of_reviews', 'review_scores_rating',
'review_scores_accuracy', 'review_scores_cleanliness',
'review_scores_checkin', 'review_scores_communication',
'review_scores_location', 'review_scores_value']

for feature_name in to_float:
    listings[feature_name] = listings[feature_name].astype(float)
    
# Remove features that are obviously erroneous entries in tht dataset
listings = listings[listings.bedrooms != 0]
listings = listings[listings.beds != 0]
listings = listings[listings.bathrooms != 0]
listings = listings[listings.accommodates != 0]
listings = listings[listings.maximum_nights != 0]

listings = listings.join(y)
listings = listings[listings.price != 0]

entries = listings.shape[0]

# Find the number of listings by neighbourhood
nbCounts = Counter(listings.neighbourhood_cleansed)
tdf = pd.DataFrame.from_dict(nbCounts, orient='index').sort_values(by=0)

ax = tdf.plot(kind='bar', figsize = (50,10), color = BNB_BLUE, alpha = 0.85)
ax.set_title("Neighborhoods by Number of Listings")
ax.set_xlabel("Neighborhood")
ax.set_ylabel("# of Listings")
#plt.show()

# Delete entries with neighbourhoods that have less than 100 entries in the dataset
for ctr in list(nbCounts):
    if nbCounts[ctr] < 100:
        del nbCounts[ctr]
        listings = listings[listings.neighbourhood_cleansed != i]

tdf = pd.DataFrame.from_dict(nb_counts, orient='index').sort_values(by=0)
ax = tdf.plot(kind='bar', figsize = (22,4), color = BNB_BLUE, alpha = 0.85)
ax.set_title("Neighborhoods by House # (Top 48)")
ax.set_xlabel("Neighborhood")
ax.set_ylabel("# of Listings")

#plt.show()

entries = listings.shape[0]

listingsClean = listings

#Create new csv to be used next
listingsClean.to_csv(path_or_buf='listings_clean.csv')

