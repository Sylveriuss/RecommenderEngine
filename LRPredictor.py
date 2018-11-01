# -*- coding: utf-8 -*-
#
# Algorithm that will train using previous ratings and predict on new values.
# It will take the already processed MoviesMetadata (.dat) to train.
#
# Warning : it is assumed that the training file (ratings.csv) has the users in sorted and in order
#

import Tools as rtools
import LinearRegressionGradientDescent as LRGR
import MovieMetadataReader as movMtdata

import csv
import time
from scipy.spatial import distance



# @RunPrediction : To Run the training on an user and making predictions for a list of movies
#-------
# user_id : the user id (integer)
# userRatedMovies : list of numpy.array and float (example [ np.array([1,2,0,1,2]), 5, np.array([1,2,7,1,2]), 0])
#                   Alternatively, there will be each instance and their ratings (x and t).
#                   In the example : x_1 = np.array([1,2,0,1,2]), t_1 = 5, x_2 = np.array([1,2,7,1,2]), t_1 = 0
# moviesToBeRated : list of movie_id that want ratings (list of strings)
# dfTF : numpy.array of the movie's feature vector
# dfIndex : dfTF's row indexes as a list of strings
# maxRate : maxRate in data (5 in our case)
# RatingType : how to compute the prediction from the userProfile (string) (see below the main function for further understanding)
# @return : (string) the predictions will be saved in the output string. (a line per prediction)
#-------
def RunPrediction(user_id, userRatedMovies, moviesToBeRated, dfTF, dfIndex, maxRate, RatingType):

    predictions = ""
    
    # Computing the Linear Regression to find the user profile
    userProfile = LRGR.LinearRegressionGradientDescent(userRatedMovies)
    
    # For every movie that needs prediction
    for movieid in moviesToBeRated:
        
        # If the movie is in our dataset
        if movieid in dfIndex:
            
            # Get the movie index in the indexes list
            movieIndex = list(dfIndex).index(movieid)
                            
            # Make prediction
            
            if RatingType == "DOTPRODUCT":                            
                pred = dfTF[movieIndex].dot(userProfile)
                
            elif RatingType == "COSINE":
                pred = distance.cosine(dfTF[movieIndex], userProfile) * maxRate
                
            elif RatingType == "BRAYCURTIS":
                pred = distance.braycurtis(dfTF[movieIndex], userProfile) * maxRate
                
            else :
                # For safety, but will not be called.
                pred = maxRate / 2.
                
            # To round the prediction at 0.5
            pred = round(pred * 2) /2
            
            # Needs a better solution, in case the value is too big
            if pred > maxRate: 
                pred = maxRate
                         
            # write this rating in the output string
            predictions += str(user_id) + "," + str(movieid) + "," + str(pred) + "\n"
            
                
        #else: 
            #print("Skipped Movie item, because not in the index : " + movieid)
            
    return predictions



""" EngineRunnerLRPred : Main Function """
# This function train the linear Regression model for each user (according to its ratings).
# From this model (userProfile), it will provide ratings for new movies.
# There are some options for the choice of the parameters and the computation of the rating.
#-------
# FileArgs: List of the paths for [TrainingDataFile, EvaluationDataFile, OutputFile] (list of string)
# FeaturesType : The parameters of the movie metadata that will be taken into account. (string)
#                "BASIC" : genres and releaseDate
#                "INTERMEDIATE" : as "BASIC", plus voteAverage and Popularity
#                "ADVANCED" : as "INTERMEDIATE", plus runtime and isAdult
#                "ALL" : as "ADVANCED", plus collection and languages
# RatingType : The computation to make prediction from the userProfile (string)
#              "DOTPRODUCT" is the normal way
#              "COSINE" is to take the cosine distance between the userProfile and the movie Feature Vecture
#              "BRAYCURTIS" is to take the bray-curtis distance (used in biology)
# log : to display the logs
# @return : the output will saved it a file (path should be indicated in FileArgs[2])
#
#
def EngineRunnerLRPred(FileArgs, FeaturesType = "BASIC", RatingType = "DOTPRODUCT", log = True):
    
    # The options of the list of parameters that can be used
    featureTypes = {"BASIC":["genre", "releaseDate"]
    , "INTERMEDIATE": ["genre", "releaseDate", "popularity", "voteAverage"]
    , "ADVANCED": ["genre", "releaseDate", "popularity", "voteAverage", "adult", "runtime"]
    , "ALL": ["genre", "releaseDate", "popularity", "voteAverage", "adult", "runtime", "collection", "language"]}
        
    
    # Checking that all the parameters have been specified
    
    if len(FileArgs) == 3 :
        trainingFile = FileArgs[0]
        testFile = FileArgs[1]
        outputFile = FileArgs[2]
    else : 
        print("Number of FileArgs not right")
        return
    
    if FeaturesType not in featureTypes: 
        print("FeaturesType not right")
        return
    
    if RatingType not in ["DOTPRODUCT", "BRAYCURTIS", "COSINE"]:
        print("DistanceType not right")
        return
    
    
    if log:
        print("----------------------------------------")
        print("Reading Movie's Metadata from .dat files")
        print("----------------------------------------")
    
    start = time.time()
    
    # Getting the MovieMetadata as matrix and its row indexes as list of strings
    dfTF, dfIndex = movMtdata.MovieMetadataRetriever(featureTypes[FeaturesType])
                                                                  
    end = time.time()
    if log:
        print( "Reading Memmap DataMatrix & index execution time : " + str(end - start)) 
    
    
    if log:
        print("-----------------------------------")
        print("Normalize Data Frame on Movie Data")
        print("-----------------------------------")    
    
    start = time.time()
    
    dfTF = rtools.normalize(dfTF)
    
    end = time.time()
    if log:
        print( "Data Normalization execution time : " + str(end - start))

    
    if log:
        print("--------------------------------")
        print("Reading Evaluation Ratings")
        print("--------------------------------")
    
    start = time.time()
    
    evalutionByUser = rtools.readCsvEvaluationData(testFile, log)
    
    end = time.time()
    if log:
        print( "Reading the evaluation file execution time : " + str(end - start))
    
    
    if log:
        print("---------------------------------------------")
        print("Reading Ratings And Online Prediction by user")
        print("---------------------------------------------")
    
    # Max value of rating 
    maxRate = 5
    
    start = time.time()
    
    # Opening the trainingFile with the previous ratings
    ifile = open(trainingFile, "r", encoding="utf8")
    reader = csv.reader(ifile, delimiter=",")
    
    rownum = 0
    errornum = 0
    
    # The first user (assuming the file has been sorted)
    last_user_id = 1    
    
    # List that we will kept the rated movies and their ratings (for each user)
    userRatedMovies = []
    
    # Opening the outputFile to write the output
    ofile = open(outputFile, "w", encoding="utf8")
    ofile.write("user_id,movie_id,ratings\n")
    
    for row in reader:
        
        try:
            
            if len(row) != 4:
                errornum += 1
                continue
        
            if row[0] == 'userId':
                continue
                    
            if log and rownum % 500000 == 0:
                print("Reading file "+ trainingFile +" : line count ... "+ str(rownum))
            
            # user_id 
            user_id = (int(row[0]))      
            
            # movie_id
            movie_id = (row[1])   
            
            # rate : The rate is divided by maxRate to normalize the value
            rate = float(row[2]) / float(maxRate)
                
            # A new user is being read ... Predict fot the last user
            if user_id != last_user_id:
                
                # Run Prediction for the last user.  		
        		  
                if len(userRatedMovies) != 0:
                                        
                    # This user is needed for the evaluation
                    if last_user_id in evalutionByUser:
                        
                        predictions = RunPrediction(last_user_id, userRatedMovies, evalutionByUser[last_user_id], dfTF, dfIndex, maxRate, RatingType)
                        
                        ofile.write(predictions)                        
                        
                    #else: 
                        #print ("User not demanded for evaluation.")
                        
                            
                # Remise à Zéro : To pass unto the new user
                last_user_id = user_id
                
                # If the rated movie is within our movie's databas
                if movie_id in dfIndex:
                    
                    # Adding the movie feature vector and its rate
                    userRatedMovies = [dfTF[list(dfIndex).index(movie_id)],rate]

            # No new user.    
            # If the rated movie is within our movie's database
            elif movie_id in dfIndex:
                
                 # Adding the movie feature vector and its rate
                 userRatedMovies += [dfTF[list(dfIndex).index(movie_id)],rate]
            
            rownum += 1
            
        except:
            errornum += 1
            pass
    
    ofile.close()
    ifile.close()
    
    if log:
        print("Finish reading file "+ trainingFile +" : number of lines : "+ str(rownum) + ': number of skipped lines : ' + str(errornum))
   
    end = time.time()
    if log:
        print( "Training and Prediction execution time : " + str(end - start))
        print("End of EngineRunnerLRPred")

