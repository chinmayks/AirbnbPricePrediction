# AirbnbPricePrediction
Project: Price and Neighborhood prediction of Airbnb Listings

DataSet: http://insideairbnb.com/get-the-data.html

City: New York city

1) Introduction:

Making use of the Inside Airbnb Project's New York city listings, we attempted to predict a listing's neighborhood and price. We started with data exploration and based on the results, proceeded with data cleaning.We extracted text based features like proximity from times square and sentiment analysis of reviews from the listings and then passed them on
as inputs along with other features to a regression model in order to predict the listing's neighborhood and price range.
Our text features were extracted using the listing's descriptions and reviews and then we used NLTK to stem
the words and identify the most commonly occurring ones.We classified these words into a small number of discrete
buckets and then represented common themes present in the data which would be important keywords that attract
attention such as comfort levels, nature of the listing, proximity to nearby city attractions, safety of the neighborhood,
etc. These buckets can then be used to identify the relative strengths of different listings which can serve as one of the
functions in identifying possible price ranges. Once features are extracted we tested it with a regression model to predict price and neighborhood.
  
2) Procedure to Pre-Process and Clean the Data

Removed attributes deemed unnecessary and known to not contribute.
Removed attributes found to be empty for > 60% of entries.
Converted required numeric fields into float type while compensating for unavailable values.

  Data Exploration:
  
    datasetExploration.py
    cleaningAndPreprocessing.py
    splitDataset.py
    createExpandedFeatureSets.py
  
2) Creation and Extraction of Features:
Main Features
  
  Property type
  Room type
  Overall rating scores
  Proximity to Times Square (the assumed centre of the city) - proximityToTimesSquare.py
  Sentiment analysis of listing descriptions and listing reviews- sentimentAnalysis.py
  
3) Models Used
   
neighbourhoodPrediction.py
  pricePrediction.py
   
   Baseline Models used:
   
      Linear Regression
      Ridge Regression
      Bayes-Ridge Regression
      Elastic Net Regularization
      
    Ensemble Model:
    
      Gradient Boosting Regressor 
      
 4) Conclusion:
 
The most infuential factors affecting price is distance from
the most preferred area or economic zone, which in our case
was Times Square and type of house: Apartment Condo /Townhouse / Loft.We even tried sentiment analysis on the
description and user reviews and tried to extract features
from it. But it didn't contribute much to price prediction.
Proximity was the most in
uential feature affecting prediction of neighborhood and house type having little contribu-
tion towards it.
For future work, Image analysis using openCV.It could have
been used to extract visual features like quality of photo
uploaded of the listings, which we think also is an essential
part in deciding price of a listing. More detailed sentiment
analysis can be done in order to lower mean absolute error.
 
      
    
 

  
