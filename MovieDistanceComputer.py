# -*- coding: utf-8 -*-
#
#   To compute the Distance between Movies.
# 


import Tools as rtools
import MovieMetadataReader as movMtdata
from scipy.spatial import distance

import time

def MoviesDistance(FeaturesType = "BASIC", outputFile = "", log = True):
    
    # The options of the list of parameters that can be used
    featureTypes = {"BASIC":["genre", "releaseDate"]
    , "INTERMEDIATE": ["genre", "releaseDate", "popularity", "voteAverage"]
    , "ADVANCED": ["genre", "releaseDate", "popularity", "voteAverage", "adult", "runtime"]
    , "ALL": ["genre", "releaseDate", "popularity", "voteAverage", "adult", "runtime", "collection", "language"]}
        
    if outputFile == "":
        print("Please specify the output file")
        return
    
    if FeaturesType not in featureTypes: 
        print("FeaturesType not right")
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
        print("---------------------------------")
        print("Write CSV of movies distances")
        print("---------------------------------")  
  
    nbMovies = len(dfIndex)
    print(nbMovies)
    with open(outputFile, "w", encoding="utf8") as ofile:
        
        ofile.write("movieId,movieId,distance\n")
        
        for i in range(nbMovies):
            
            for j in range(i):
                    
                dist = distance.cosine(dfTF[i,:], dfTF[j,:])
                ofile.write(dfIndex[i] + "," + dfIndex[j] + "," + str(dist) + "\n")
                    
MoviesDistance("INTERMEDIATE","movie_distance_2.csv")