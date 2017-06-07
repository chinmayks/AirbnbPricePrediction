# AirbnbPricePrediction
Project: Price and Neighborhood prediction of Airbnb Listings

DataSet: http://insideairbnb.com/get-the-data.html

City: New York city

1) Introduction:
Making use of the Inside Airbnb Project's New York city listings, we attempted to predict a listing's neighborhood and price. We started with data exploration and based on the results,
proceeded with data cleaning.We extracted text based fea-
tures like proximity from times square and sentiment anal-
ysis of reviews from the listings and then passed them on
as inputs along with other features to a regression model in
order to predict the listing's neighborhood and price range.
Our text features were extracted using the listing's de-
scriptions and reviews and then we used NLTK to stem
the words and identify the most commonly occurring ones.
We classified these words into a small number of discrete
buckets and then represented common themes present in
the data which would be important keywords that attract
attention such as comfort levels, nature of the listing, proximity to nearby city attractions, safety of the neighborhood,
etc. These buckets can then be used to identify the relative
strengths of different listings which can serve as one of the
functions in identifying possible price ranges. Once features
are extracted we tested it with a regression model to predict
price and neighborhood.
  
2) Procedure to Pre-Process and Clean the Data

Removed attributes deemed unnecessary and known to not contribute
Removed attributes found to be empty for > 60% of entries
Converted required numeric fields into float type while compensating for unavailable values

  Data Exploration:
  
    datasetExploration.py
    cleaningAndPreprocessing.py
    splitDataset.py
    createExpandedFeatureSets.py
  
2) Creation and Extraction of Features
  
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
      
    
 

  
