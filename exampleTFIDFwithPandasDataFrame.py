# -*- coding: utf-8 -*-
#
# TF-IDF 
# My first approach to solve this problem of Recommendation
# My code to test a simple use case (as in https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/#)
# For our data, this isn't the best solution. Still it is an interesting technique.
#

import pandas as pd
import numpy as np
from scipy.spatial import distance


def computingTF(df_, binaryRepresentation = False):
    if not binaryRepresentation:
        res = np.zeros(df_.shape)
        for i in df_.index:
            for j in range(df_.shape[1]):
                val = df_.loc[i].iloc[j]
                res[list(df_.index).index(i)][j] = (1 + np.log10(val) if int(val) != 0 else 0 )
        return res
    else:
        return df_.values

def computeIDF(df_, nb):
    res = np.zeros((1,df_.shape[1]))
    for i in range(df_.shape[1]):
        val = df_.iloc[:,i]
        sumVal = sum(val) 
        res[0][i] = (np.log10(nb / sumVal) if sumVal != 0 else 0)
    return res


def normalize(df_):
    res = np.zeros(df_.shape)
    for i in range(df_.shape[0]):
        iLoc = np.array(df_[i,:])
        lengthVector = np.sqrt((iLoc.dot(iLoc)).sum())
        for j in range(df_.shape[1]):
            val = df_[i,j]
            res[i,j] = val / lengthVector
    return res


def cosineDistance(df_):
    nb = df_.shape[0]
    res = np.zeros((nb,nb))
    for i in range(nb):
        for j in range(nb):
            res[i,j] = distance.cosine(df_[i,:], df_[j,:])
    return res

def userProfileMaker(userRating, dfTFNormalised):
    userProfileMaker = np.zeros((len(userRating.keys()), dfTFNormalised.shape[1]))
    userID = 0
    for user in userRating:
        for element in range(dfTFNormalised.shape[1]):
            val = userRating[user].dot(dfTFNormalised[:,element])
            userProfileMaker[userID,element] = val
        userID += 1
    return userProfileMaker

def makePred(movie, user, idf):
    res = 0
    for i in range(len(movie)):
        res += movie[i] * user[i] * abs(idf[0][i])
    return res

def testMain():
    
    print ("TEST TF-IDF")
    
    testMatrixDict = {
        "analytics": [1,0,0,0,0,1],
        "r": [0,1,0,0,1,0],
        "python": [1,1,0,1,0,0],
        "machineLearning": [0,1,1,1,0,1],
        "learningpaths": [1,0,1,0,0,0]
    }
    testMatrixDF = pd.DataFrame(testMatrixDict, columns=list(testMatrixDict.keys()))
    
    normMatrixDict = {
        "analytics": [0.57735027,0,0,0,0,0.70710678],
        "r": [0,0.57735027,0,0,1,0],
        "python": [0.57735027,0.57735027,0,0.70710678,0,0],
        "machineLearning": [0,0.57735027,0.70710678,0.70710678,0,0.70710678],
        "learningpaths": [0.57735027,0,0.70710678,0,0,0]
    }
    normMatrixDF = pd.DataFrame(normMatrixDict, columns=list(normMatrixDict.keys()))
    
    distMatrix = cosineDistance(np.transpose(normMatrixDF.values))
    
    #print("the document 'analytics' is "+ str(round(distMatrix[0,3] * 100)) + "% distant for the Document 'machineLearning'.")
    #print("the document 'python' is "+ str(round(distMatrix[2,3] * 100)) + "% distant for the Document 'machineLearning'.")
    
    assert str(round(distMatrix[0,3] * 100)) > str(round(distMatrix[2,3] * 100))
    
    testMatrixIDF = computeIDF(testMatrixDF, 10)
    
    normTest = normalize(computingTF(testMatrixDF,True))
    
    testRatings = {
        "one": [1,-1,0,0,0,1],
        "two": [-1,1,0,1,0,0]
    }
    testRatingsDF = pd.DataFrame(testRatings, columns=list(testRatings.keys()))
    
    testProfile = userProfileMaker(testRatingsDF, normTest)
    
    assert makePred(normTest[0], testProfile[0],testMatrixIDF) == 0.7513333122463779
    
    print ("TEST Finished : Success")
    
testMain()