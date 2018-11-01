# -*- coding: utf-8 -*-

import Tools as rtools
import LRPredictor
import MovieMetadataReader as movieMdat
   
import sys

    
    
#--------
# To Run the Algorithm on differents configurations of Data and differents ways of predictions
# To Evaluate the results with the RMSE metric
#--------
# trainFile : path to file that has the previous ratings (string)
# testFile : path to file that has the couple (userId, movieID) to be rated (string)
# testTargetFile : path to the file that has the right ratings, for the couple mentionned in the testFile (string)
# featureTypes : list of parametres configuration to be used from Movie Metadata Matrix (list of strings)
# ratingTypes : list of rating's prediction computation to be done (list of strings)
# @return : (void) && print the dict containing the rmse for each configurations.
#--------
def Evaluate(trainFile, testFile, testTargetFile, featureTypes = ["INTERMEDIATE"], ratingTypes = ["DOTPRODUCT", "COSINE", "BRAYCURTIS"]):
    
    res = {}
    
    for featureType in featureTypes :
    
        for ratingType in ratingTypes:
            
            resultFile = featureType+"_"+ratingType+"_Evaluate.csv"
            
            LRPredictor.EngineRunnerLRPred([trainFile, testFile, resultFile], featureType, ratingType, False)
            res[featureType+"_"+ratingType] = rtools.RMSEeval(testTargetFile, resultFile)            
            
    print("Result (rmse) :")
    print(res)
        

#--------
# To Run the algorithme on a specific configuration
#--------
# trainFile : path to file that has the previous ratings (string)
# testFile : path to file that has the couple (userId, movieID) to be rated (string)
# resultFile : path to file that will contain the output
# featureTypes : list of parametres configuration to be used from Movie Metadata Matrix (list of strings)
# ratingTypes : list of rating's prediction computation to be done (list of strings)
# @return : (void) The output will generated in a file (resultFile)
#--------
def RunPredictor(trainFile, testFile, resultFile, featureType = "INTERMEDIATE", ratingType = "DOTPRODUCT"):
    
    if resultFile == "":
        resultFile = featureType+"_"+ratingType+"_Run.csv"
    
    LRPredictor.EngineRunnerLRPred([trainFile, testFile, resultFile], featureType, ratingType, False)
    
    print("The Run has Finished. The output is in the file : "+resultFile+".")


#--------
# Main Function
#--------
def Main():
    
    # Arguments
    if len(sys.argv) < 2:
        print("Need arguments : > Main.py [InputDatasDirectory with /] [PathOutputFile] ")
        return
    
    dataDirectory = sys.argv[0]
    outputFile = sys.argv[1]
    
    # Process the MetaData of the movies
    movieMdat.MovieMetadataProcessor(dataDirectory+'movies_metadata.csv')
    
    # The prediction Algorithm
    RunPredictor(dataDirectory+"ratings.csv", dataDirectory+"evaluation_ratings.csv", outputFile, "INTERMEDIATE", "DOTPRODUCT")
    
    # Clean the Data
    movieMdat.cleaner()

Main()