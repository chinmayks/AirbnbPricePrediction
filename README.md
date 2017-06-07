# AirbnbPricePrediction
Project: Price and Neighborhood prediction of Airbnb Listings
DataSet: http://insideairbnb.com/get-the-data.html
City: New York city

1) Procedure to Pre-Process and Clean the Data (cleaningAndPreprocessing.py)

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
      
    
 

  
