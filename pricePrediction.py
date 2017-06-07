"""
pricePrediction.py
authors: Akshay Sharma , Chinmay Singh
description: Finds how well we can predict the neighbourhood using the features extracted and then determines the most effective feature
"""

from sklearn import ensemble
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics

training = pd.read_csv('features_expanded.csv')
testing = pd.read_csv('features_expanded_testing.csv')

training = training.fillna(method='ffill')
testing = testing.fillna(method='ffill')

#Hot encode categorical data
nDummies = pd.get_dummies(training.neighbourhood)
rtDummies = pd.getdummies(training.room_type)
ptDummies = pd.get_dummies(training.property_type)

#Create dataset with dummy data
completeDataTraining = pd.concat((training.drop(['neighbourhood', 'room_type', 'property_type'], axis=1),
                          nDummies.astype(int), rtDummies.astype(int), ptDummies.astype(int)), axis=1)

#Prep the models
estimators = [ linear_model.LinearRegression(), linear_model.Ridge(), linear_model.ElasticNet(),
        linear_model.BayesianRidge()]

labels = np.array(['Linear', 'Ridge','ElasticNet', 'BayesRidge'])
errorValues = np.array([])

#Hot encode categorical data
nDummies = pd.get_dummies(testing.neighbourhood)
rtDummies = pd.getdummies(testing.room_type)
ptDummies = pd.get_dummies(testing.property_type)

#Create dataset with dummy data
completeDataTesting = pd.concat((testing.drop(['neighbourhood', 'room_type', 'property_type'], axis=1),
                          nDummies.astype(int), rtDummies.astype(int), ptDummies.astype(int)), axis=1)

#Create predictor and response sets
X = ['polarities', 'review_scores_rating',
       'distance_from_times_square', 'Astoria',
       'Bedford-Stuyvesant', 'Boerum Hill', 'Brooklyn Heights', 'Bushwick',
       'Carroll Gardens', 'Chelsea', 'Chinatown', 'Clinton Hill',
       'Crown Heights', 'Ditmars Steinway', 'East Flatbush', 'East Harlem',
       'East New York', 'East Village', 'Elmhurst', 'Financial District',
       'Flatbush', 'Flushing', 'Fort Greene', 'Gowanus', 'Gramercy',
       'Greenpoint', 'Greenwich Village', 'Harlem', 'Hell''s Kitchen', 'Inwood',
       'Jackson Heights', 'Kensington', 'Kips Bay', 'Long Island City',
       'Lower East Side', 'Midtown', 'Morningside Heights', 'Murray Hill',
       'Nolita', 'Park Slope', 'Prospect Heights', 'Prospect-Lefferts Gardens',
       'Real Bed', 'Ridgewood', 'Sheepshead Bay', 'SoHo', 'South Slope',
       'Sunnyside', 'Sunset Park', 'Theater District', 'Tribeca',
       'Upper East Side', 'Upper West Side', 'Washington Heights',
       'West Village', 'Williamsburg', 'Windsor Terrace', 'Woodside', '8.0',
       'Entire home/apt', 'Private room', 'Shared room', '3 weeks ago',
       'Apartment', 'Bed & Breakfast', 'Boat', 'Boutique hotel', 'Bungalow',
       'Cabin', 'Castle', 'Cave', 'Chalet', 'Condominium', 'Dorm',
       'Guesthouse', 'Hostel', 'House', 'Hut', 'Lighthouse', 'Loft', 'Other',
       'Serviced apartment', 'Tent', 'Timeshare', 'Townhouse', 'Villa']

Y = ['price']

#Train the models
for estimator in estimators:
        estimator.fit(completeDataTraining[X], completeDataTraining[y])
        curError = metrics.median_absolute_error(completeDataTesting[y], estimator.predict(completeDataTesting[X]))
        errorValues = np.append(errorValues, curError)
        
possibilities = np.arrange(errorValues.shape[0])
srted = np.argsort(errorValues)
plt.figure(figsize=(7,5))
plt.bar(possibilities, errorValues[srted])
plt.xticks(possibilities, labels[srted])
plt.xlabel('Estimator Type')
plt.ylabel('Median Absolute Error')

# plt.show()

#Use gradient boost to attempt to improve predictions
numEstimators = 250
tuningParameters = {
        "n_estimators":[numEstimators],
        "max_depth":[7],
        "learning_rate":[0.01],
        "min_samples_split":[1.00],
        "loss":['ls','lad']                    
        }

regressor = ensemble.GradiantBoostingRegressor()
clf = GridSearchCV(regressor, cv=3, param_grid=tuningParameters, scoring='neg_median_absolute_error')
preds = clf.fit(completeDataTraining[X], completeDataTraining[y])
best = clf.best_estimator_

print(abs(clf.best_score_))

test_score = np.zeros(numEstimators, dtype=np.float64)

train_score = best.train_score_
for i, prediction in enumerate(best.staged_predict(completeDataTesting[X])):
    test_score[i] = metrics.median_absolute_error(completeDataTesting[y], prediction)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(numEstimators), train_score, 'darkblue', label='Training Set Error')
plt.plot(np.arange(numEstimators), test_score, 'red', label='Test Set Error')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Median Absolute Error')

featureImportance = clf.best_estimator_.feature_importances_
featureImportance = 100.0*(featureImportance/featureImportance.max())
srted = np.argsort(featureImportance)
pos = np.arange(srted.shape[0]) + 0.5
values = featureImportance[srted]
cols = completeDataTraining[srted]
plt.figure(figsize = (10,12))
plt.barh(pos, values, align='center')
plt.yticks(pos, cols)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')

plt.show()