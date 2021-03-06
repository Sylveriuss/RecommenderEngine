# -*- coding: utf-8 -*-
#
# Data Processing of the file movies_metadata.csv
# Return the result in files.
#

import Tools as rtools
import env
import csv
import pandas as pd
import numpy as np
import time
import os


""" @MovieMetadataProcessor:  Main Function that process the data """
# This function process the parameters of the metadatas of movies.
#-------
# FileArg : path to the metadata_movie CSV File. (string)
# OutpurDir : path to the directory for the output files with / at the end. (string)
# Log : to display the logs
# @return :
#       - Create a movieDF.dat file : the numpy array that has been processed. 
#       - Create a movieIndex.dat file : the index of the rows of movieDF (movie_id as integers)
#       - Create a moviesColumns.csv file : the header of the columns of movieDF
#
# In the process chosen: 
#       Identification : 'movie_id' as main Id
#       To keep the values -> 'adult' 'collection' 'genres' 'popularity' 'release_date' (only the year)
#                             'runtime' 'languages' (spoken and original as distinct list) 'vote_average'
#
#
#
def MovieMetadataProcessor(FileArg, OutpurDir = "", Log = False):
        
    if FileArg != "":
        metadataFile = FileArg
    else : 
        print("Should indicate the path to the movie metadata file.")
        return
    
    if Log:
        print("--------------------------------")
        print("Reading Movie's Metadata")
        print("--------------------------------")
    
    # Reading Csv in list a
    a = rtools.readcsv(metadataFile)    
    
    
    if Log:
        print("--------------------------------")
        print("Cleaning MovieData")
        print("--------------------------------")
    
    # Dict that will contain the values of the selected parameters.    
    instances = {
        'adult': [],
        'collection': [],
        'genres': [],
        'movie_id': [],
        'popularity': [],
        'release_date': [],
        'runtime': [],
        'languages': [],
        'vote_average': []
    }
    
    # The LookUp lists for Categorical Features (Not used)
    collectionLookUp = {}
    genresLookUp = {}
    languagesLookUp = {}
    
    start = time.time()
    
    # Getting the data from list to the dict
    rtools.getData(a, instances, collectionLookUp, genresLookUp, languagesLookUp)
    
    end = time.time()
    
    if Log:
        print( "getData() execution time : " + str(end - start))
    
    # Cleaning ... 
    rtools.free(a)
    rtools.free(collectionLookUp)
    rtools.free(genresLookUp)
    rtools.free(languagesLookUp)
    
    # Asserting so that the rest will be coherent
    assert rtools.verifyNbInstancesAlign(instances)
    
    nbinstances = len(instances['movie_id'])
    
    if Log:
        print("Nomber of movie instances : " + str(nbinstances))
    
    # Parameter 'Popularity' : 
    #   Since some values were 'too high' -> replacement of those greater the 99.95 percentile by the mean.
    #   To categorize -> In four equal parts according to the percentiles.
    keptPopularity = rtools.removingInsecureValues(instances['popularity'], 99.95, 0)
    popularityList = rtools.determineCategoriesBoudaries(keptPopularity, 4)
    popularityVect = rtools.categorizeVectContinuousVar(keptPopularity, popularityList)
    
    # Parameter 'Release_Date' : 
    #   Since some values were equal to 0 -> replacement of those values by the mean.
    #   To categorize -> In ten equal parts according to the percentiles.
    keptDates = rtools.removingNullValues(instances['release_date'], True)
    dateList = rtools.determineCategoriesBoudaries(keptDates, 10)
    releaseDatesVect = rtools.categorizeVectContinuousVar(keptDates, dateList)
    
    # Parameter 'Vote_Average' : 
    #   To categorize -> In five parts with the inner boundaries as [2.5, 5.0, 6.125, 7.5].
    keptAverages = instances['vote_average']
    voteAveragesVect = rtools.categorizeVectContinuousVar(keptAverages, [2.5, 5.0, 6.125, 7.5])
    
    # Parameter 'Runtime' : 
    #   To categorize -> In 3 parts with the inner boundaries as [60,180].
    keptRuntime = instances['runtime']
    runtimeVect = rtools.categorizeVectContinuousVar(keptRuntime, [60,180])
    
    # Parameter 'Adult' :
    keptAdult = instances['adult']    
    adultVect = keptAdult
    
    # Cleaning ... 
    rtools.free(keptDates)
    rtools.free(keptPopularity)
    rtools.free(keptAverages)
    rtools.free(keptRuntime)
    rtools.free(keptAdult)
    
    # Parameter 'Collection' :
    #   Since the list is wide -> keeping only frequent collections at 80%.
    collectionList = rtools.getListOfRelevantItemInFeatures(instances['collection'], 80)
    collectionVect = rtools.categorizeVectVar(instances['collection'], collectionList)
    
    # Parameter 'Genres' :
    genresList = [ x for x in list(genresLookUp.keys()) if x != 0]
    genresVect = rtools.categorizeVectVar(instances['genres'], genresList)
    
    # Parameter 'Languages' :
    languagesList = [ x for x in list(languagesLookUp.keys()) if x != 0]
    languagesVect = rtools.categorizeVectVar(instances['languages'], languagesList)
    
    
    ## Combining each parameters : Data and Columns
    ## The order must be respected
    
    data = [instances['movie_id'], popularityVect, releaseDatesVect, voteAveragesVect, runtimeVect, adultVect
            , collectionVect, genresVect, languagesVect]
    
    # The list of parameters : Continuous and Categorical
    continuousData = [
        ['popularity', popularityList]
        , ['releaseDate', dateList]
        , ['voteAverage', [2.5, 5.0, 6.125, 7.5]]
        , ['runtime', [60,180]]
    ]
    categoricalData = [
        ['adult',['isAdult'], False]
        , ['collection',collectionList, True]
        , ['genre',genresList, True]
        , ['language',languagesList, True]
    ]
    columnsDF = ['movie_id'] + rtools.formingColumnsDF(continuousData, categoricalData)
    
    # Writing the columns (without the movie_id column)
    with open(OutpurDir + env.MMDT_COLUMNS, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in columnsDF[1:]:
            writer.writerow([val])
    
    # Cleaning ...
    rtools.free(popularityList)
    rtools.free(popularityVect)
    rtools.free(dateList)
    rtools.free(releaseDatesVect)
    rtools.free(voteAveragesVect)
    rtools.free(runtimeVect)
    rtools.free(adultVect)
    rtools.free(collectionVect)
    rtools.free(genresVect)
    rtools.free(languagesVect)
    rtools.free(collectionList)
    rtools.free(genresList)
    rtools.free(languagesList)
    
    # Asserting so that the rest will be coherent (That we haven't lost any instance)
    assert rtools.verifyNbInstancesList(data,nbinstances)
    
    # Having a list of row (instances)
    newData = rtools.formingTheDictForDF(data)
    
    # Asserting so that the rest will be coherent
    assert len(newData) == nbinstances
    assert len(newData[0]) == len(columnsDF)
    
    ## Making a DataFrame out it. (Historically, wanted to work to DataFrames, 
    ##  but it turned out that numpy.array were much better for my case.)
    df = pd.DataFrame(newData, columns=columnsDF)
    
    rtools.free(data)
    rtools.free(newData)
    rtools.free(columnsDF)
    
    df.set_index('movie_id', inplace=True)
    
    if Log:
        print("----------------------------------")
        print("Registering the Data as .dat files")
        print("----------------------------------")
        
    # Registering the Indexes of the Data Frame as Integer values (.dat) for futher use
    dfIndex = df.index
    dfIndexAsInt = [int(elem) for elem in dfIndex]
    indexMemmap = np.memmap(OutpurDir + env.MMDT_ROWINDEX, dtype='int64', mode='w+', shape=dfIndex.shape)
    indexMemmap[:] = dfIndexAsInt[:]

    # Registering the Data Frame as Numpy.Array .dat
    dfTF = df.values    
    dfMemmap = np.memmap(OutpurDir + env.MMDT_DATAFRAME, dtype='float32', mode='w+', shape=dfTF.shape)
    dfMemmap[:] = dfTF[:]
    
    
    
    
""" MovieMetadataRetriever : Function to get the data from the registered files """
#
# neededColumns: list of string indicating the columns (parameters) of the data that will be kept.
#                example: ["genre", "releaseDate", "popularity", "voteAverage"]
#------
# It will return the data as a numpy.array and the rowIndex as a list of strings
# @return: dfTF (numpy.array), dfIndex (list of strings)
#------
def MovieMetadataRetriever(neededColumns):    
    
    # Getting the Indexes of the rows 'Movie_id'
    dfTFIndex = np.memmap(env.MMDT_ROWINDEX, dtype='int64', mode='r')
    dfIndex = [str(elem) for elem in dfTFIndex]
    
    # Getting the Data as Numpy.Array
    dfTF = np.memmap(env.MMDT_DATAFRAME, dtype='float32', mode='r')
    
    # Reshaping as matrix ( Num_Movie_ids x Num_Parameters )
    dfTF = dfTF.reshape((len(dfTFIndex), int(len(dfTF)/len(dfTFIndex))))
    
    # Getting the Columns of the data matrix
    dfColumns = []
    with open(env.MMDT_COLUMNS, "r") as inputColumns:        
        index = 0        
        for val in inputColumns:
            
            # keeping only the selected columns from the input neededColumns
            if val.split('_')[0] in neededColumns:
                dfColumns += [index]
                
            index += 1
    
    # From the Data matrix, getting only the wanted columns    
    dfTF = dfTF[:, dfColumns]
    
    return dfTF, dfIndex

# @cleaner : Remove create dat files
#---------
def cleaner():
    os.remove(env.MMDT_ROWINDEX)
    os.remove(env.MMDT_DATAFRAME)
    os.remove(env.MMDT_COLUMNS)
    
#MovieMetadataProcessor('movies_metadata.csv')
#MovieMetadataRetriever(["genre", "releaseDate"])