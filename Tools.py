# -*- coding: utf-8 -*-
#
# Tools contains many of the functions needed to process the movies_metadata.csv file, the evaluation_ratings.csv file.
# And other functions ...like to normalize, to compute the cosine distance.
#

import csv
import numpy as np
import statistics
import scipy.stats as stats
from collections import Counter
from sklearn.metrics.pairwise import cosine_distances

import env


# @free : to Delete variable
#--------
# var : is of any type.
# @return : (void)
#--------
def free(var):
	del var



# @readcsv : to Read a csv file
#--------
# filename : is full path or relative to the current directory (string).
# res : is a list, not necessarily empty.
# @return : res or a new list that will contain the lines of read file. (list)
#--------
def readcsv(filename, res = []):
    ifile = open(filename, "r", encoding="utf8")
    reader = csv.reader(ifile, delimiter=",")

    rownum = 0
    errornum = 0

    for row in reader:
        
        try:
            
            # To write down the progression
            if rownum % 50000 == 0:
                print("Reading file "+ filename +" : line count ... "+ str(rownum))
            
            res.append(row)
            
            rownum += 1
        
        except:
            # Skipped Rows
            errornum += 1
            pass
    
    ifile.close()
    
    print("Finish reading file "+ filename +" : number of lines : "+ str(rownum))
    print('Number of skipped lines : ' + str(errornum))
    
    return res

	

# @getElemFromJson : To get value from JsonValue in the Movie metadate file 
# while remembering new values in a lookUpList
#--------
# json : the input text (string).
# idLabel : the id in the json of value you want to retrieve (string).
# nameLabel : the name in the json of value you want to retrieve (string).
# lookUpList : dict that will contain the value index and its value (dict).
# intVal : if the value you want to retrieve is an integer (bool).
# @return : the list of the values corresponding to idLabel, lookUpList will be updated with new values
#--------
# Example
# json : "[{'id': 35, 'name': 'Comedy'}, {'id':10751, 'name':'Family'}]"
# idLabel : 'id'
# nameLabel : 'name'
# lookUpList : dict{ 10749:'Romance'}
# intVal : True (the value of id is an integer)
# @return : [35, 10751] (list) && lookUpList updated : dict{ 10749:'Romance', 35:'Comedy', 10751:'Family' }
#--------
def getElemFromJson(json, idLabel, nameLabel, lookUpList, intVal):
    
    res = []
    
    # First Split to separate differents values in case it is a list
    elem = json.split("}")
    
    for e in elem:
        
        # Default values
        id_ = 0 if intVal else ""  
        name = ""
        
        # Second Split to separete attributs
        item = e.split(",")
        
        for i in item:
            
            # Removing unwanted special characters
            i = i.replace(',', '').replace('{', '').replace('}', '').replace("'", '').replace("[", '').replace("]", '')
            
            # Third Split to separate the key for its value
            s = i.split(":")
            
            if (len(s) > 1) :
                
                # If it is the key we search (assuming that there will be no problem)
                if idLabel in s[0]:
                    
                    if (intVal):
                        # if it is an integer value
                        id_ = int(s[1].strip())
                    else:
                        id_ = s[1].strip()
                        
                    res.append(id_)
                    
                # If it is the name of value we search
                if nameLabel in s[0]:
                    name = s[1].strip()
        
        # updating the LookUpList
        lookUpList[id_] = name
        
    return res

if env.TESTMODE:
    lookUp = { 10749:'Romance'}
    assert [35, 10751] == getElemFromJson("[{'id': 35, 'name': 'Comedy'}, {'id':10751, 'name':'Family'}]", 'id', 'name', lookUp, True)
    assert lookUp[35] == 'Comedy'
    assert lookUp[10751] == 'Family'



# @getData : To get the data from the list (as from readcsv) and fill the new dict that will contain in order the data.
# while remembering new values in lookUp list for 'collection', 'genres', 'languages'
#--------
# listinput: list of lists that will contain in order the data from movies_metadata.csv (list)
# instances: dict that have the following keys 'adult', 'collection', 'genres', 'movie_id',
#           'popularity', 'release_date', 'runtime', 'languages', 'vote_average' (fields that are being used for the analysis) (dict)
# collectionLookUp: dict that will contained the association key/value for every collection in the data (dict)
# genresLookUp: dict that will contained the association key/value for every genre in the data (dict)
# languagesLookUp: dict that will contained the association key/value for every language in the data (dict)
# @return : the input values instances, collectionLookUp, genresLookUp, languagesLookUp will be updated (void)
#--------
def getData(listinput, instances, collectionLookUp, genresLookUp, languagesLookUp):

    # range from 1 to skip the header
    for i in range(1, len(listinput)-1):
        inst = listinput[i]
        
        if len(inst) != 24: # If there is a skipped field in the data
            continue
        
        if inst[5] not in instances['movie_id']: # To include only one per id

            # adult
            instances['adult'].append(0 if  inst[0] == 'False' else 1)
        
            # collection : get the list of collection's id
            instances['collection'].append(getElemFromJson(inst[1], "id", "name", collectionLookUp, True)) 
        
            # genres : get the list of genres' is
            instances['genres'].append(getElemFromJson(inst[3], "id", "name", genresLookUp, True))      
        
            # movie_id
            instances['movie_id'].append(inst[5])
                
            # popularity
            instances['popularity'].append(float(inst[10]))
        
            # release_date : only the year
            instances['release_date'].append(0.0 if inst[14] == "" else float(inst[14].split("-")[0]))
        
            # runtime
            instances['runtime'].append(0.0 if inst[16] == "" else float(inst[16]))
        
            # languages : original_languages + sopken_languages : distinct list
            instances['languages'].append(list(set([inst[7]] + getElemFromJson(inst[17], "iso_639_1", "name", languagesLookUp, False)))) 
        
            # revenue        
            instances['vote_average'].append(float(inst[22]))



# @verifyNbInstancesAlign : to check if the function getData() have returned a coherent dict
#--------
# instances : dict that will contain per key lists of values (dict)
# @return : True if the length of all the list values have the same length, False otherwise (boolean)
#--------
def verifyNbInstancesAlign(instances):

    nbinstances = len(instances['movie_id'])

    for key in list(instances.keys()):
        if len(instances[key]) != nbinstances:
            return False

    return True
	


# @removingInsecureValues : to remove values that are too big ( > percentileMax) or too small ( < percentileMin) and replace them with the mean
#--------
# vector: list of real values (list)
# percentileMax: percentile that will upper bound the values (int)
# percentileMin: percentile that will lower bound the values (int)
# name: Name of the field corresponding to the values (string)
# log: to print log (boolean)
# @return : the new list with values with the boundaries (list)
#--------
# COMMENT : the mean should be calculated within the boundaries
#--------
def removingInsecureValues(vector, percentileMax = 90, percentileMin = 0, name = "", log = False):

    mean = np.mean(vector)
    maxVariance = stats.scoreatpercentile(vector, percentileMax) # Upper bound
    minVariance = stats.scoreatpercentile(vector, percentileMin) # Lower bound
    
    newVector = []

    # To indicate the changes that have been done
    nbChanges = []

    for x in vector:

        if (x > maxVariance or x < minVariance):
            # Out of bounds, replace with the mean
            newVector.append(mean)
            nbChanges.append(x)

        else : 
            newVector.append(x)
    
    if log:
        print("--------------------------------- " + str(name))
        print("removingInsecureValues : ")
        print("The mean is " + str(mean))
        print("The variance is " + str(statistics.stdev(vector) * statistics.stdev(vector))) # there must be a function for it
        print("For percentileMin = " + str(percentileMin) + ", the value is " + str(minVariance))
        print("For percentileMax = " + str(percentileMax) + ", the value is " + str(maxVariance))
        print("The number of changes in the vector is " + str(len(nbChanges)))
        print(nbChanges)
        print("")
    
    return newVector

if env.TESTMODE:
    assert removingInsecureValues([1,40,42,45,43,41,53,35,43,44,50,100],95,5) == [44.75, 40, 42, 45, 43, 41, 53, 35, 43, 44, 50, 44.75]



# @removingNullValues : to replace null values (= 0) with the mean
#--------
# vector: list of real values (list)
# intValue: if vector is of Integer values rather that Floats (boolean)
# name: Name of the field corresponding to the values (string)
# log: to print log (boolean)
# @return : the new list with no more null values (list)
#--------
# COMMENT : the mean should be calculated after the remmoval of null values
#--------
def removingNullValues(vector, intValue=True, name = "", log = False):

    mean = np.mean(vector)

    # if it is a vector of int, cast the mean to int
    if intValue:
        mean = int(mean)
        
    newVector = []

    # to indicate the number of changes
    nbChanges = 0

    for x in vector:

        if x == 0:
            # Null value, replace it with the mean
            newVector.append(mean)
            nbChanges += 1

        else : 
            newVector.append(x)
    
    if log:
        print("--------------------------------- " + str(name))
        print("removingNullValues : Mean")       
        print("The mean is " + str(mean))
        print("The number of changes in the vector is " + str(nbChanges))
        print("")
    
    return newVector

if env.TESTMODE:
    assert removingNullValues([0,40,42,45,43,41,53,35,43,44,50,0],True) == [36, 40, 42, 45, 43, 41, 53, 35, 43, 44, 50, 36]



# @categorizeVectContinuousVar : to make categorical features out of continuous features
#--------
# vector: values of the continuous feature (list of numbers)
# categories: list of the inner boundaries that will characterize each new categorie (list of numbers)
# name: Name of the field corresponding to the values (string)
# log: to print log (boolean)
# @return : the new list with categorical values. The categories are represented with [feature vectors]. (list of list of binary numbers)
#--------
# For further understanding, see example below.
#--------
def categorizeVectContinuousVar(vector, categories, name = '', log = False):
    
    wholeNewVector = []
    
    # to indicate the number of items per categories
    countingFreq = np.zeros(len(categories)+1)
    
    for elem in vector:
        
        # A [feature vector] for each item
        newVector = []
        
        # An item belongs to only one category
        foundCat = False
        
        # Loop over the bounds
        for cat in categories:
            
            if elem < cat and not foundCat:
                
                # If within the bound and have not been already marked
                newVector.append(1)
                foundCat = True     
                
            else: 
                newVector.append(0)
                
        # Last Category : greater than the last inner bound
        newVector.append(0 if foundCat else 1)
        
        # Add each vector of binary numbers gives us the number of items per category.
        countingFreq = np.add(countingFreq, newVector)
        wholeNewVector.append(newVector)
    
    if log:
        print("--------------------------------- " + str(name))
        print("categorizeVectContinuousVar : Vectorizing")       
        print("The categories inner borders are :")
        print(categories)
        print("The countingFreq is :")
        print(countingFreq)
        print("The sum of countingFreq is :" + str(sum(countingFreq)))
        print("")
        
    return wholeNewVector

if env.TESTMODE:
    # The inner bound is 2. There is then two categories : <2 and >=2. 
    # the value 3 is in the second category, so its [feature vector] is [0, 1]
    assert categorizeVectContinuousVar([1,2,3], [2]) == [[1, 0], [0, 1], [0, 1]]



# @determineCategoriesBoudaries : To determine the inner boundaries of a real valued feature
#                                  according to equal parts and percentiles
#                                   (for the need to split a real valued vector)
#--------
# vector: values of the continuous feature (list of numbers)
# nbEquiCategories: number of parts that will divide the vector (integer)
# name: Name of the field corresponding to the values (string)
# log: to print log (boolean)
# @return : the  list with inner boundaries (list of numbers)
#--------
def determineCategoriesBoudaries(vector, nbEquiCategories = 4, name = '', log = False):
    
    boundaries = []
    percentage = 100 / nbEquiCategories
    
    for part in range(1, nbEquiCategories):
        bound = stats.scoreatpercentile(vector, percentage * part)
        boundaries.append(bound)
        
    if log:
        print("--------------------------------- " + str(name))
        print("determineCategoriesBoudaries : Boudaries")
        print(boundaries)
        
    return boundaries

if env.TESTMODE:
    assert determineCategoriesBoudaries([1,2,3,4,5,6],3) == [2.666666666666667, 4.333333333333334]



# @getListOfRelevantItemInFeatures: In categorical features, to keep only the frequent items.
#                                   This is specific for the 'collection' feature of the data.
#--------
# featureDate : the categorigal featured vector (list of list of integers). An item has a list of categories ('collection').
# percentage : Percentage of frequents categories that will be kept (needs a better formulation)
# name: Name of the field corresponding to the values (string)
# log: to print log (boolean)
# @return : the list of categories that will be kept (list of integers)
#--------
def getListOfRelevantItemInFeatures(featureData, percentage = 75, name = "", log = False):
    
    listData = []
    
    # To count empty items
    nbNull = 0
    
    for elem in featureData:
        
        nbNull += 1 if elem == [] else 0
        
        # To concatenate all values
        listData += elem

    # Count the frequency of items (dict)
    freqData = Counter(listData).values()
    
    # To Get the percentile of the latter list according to the input percentage
    percentileValue = stats.scoreatpercentile(list(freqData), percentage)
    
    # To keep only the values that are under the percentile
    filterFreqData = {k:v for (k,v) in Counter(listData).items() if v > percentileValue}
    
    # To get the items' ids only
    newItems = list(filterFreqData.keys())
    
    if log:
        print("--------------------------------- " + str(name))
        print("getListOfRelevantItemInFeatures : Frequency Vector")
        print("The number of null values is " + str(nbNull))        
        print("The mean is " + str(np.mean(list(freqData))))
        print("The median is " + str(np.median(list(freqData))))       
        print("For percentage = " + str(percentage) + ", the value is " + str(percentileValue))
        print("The number of items in the frequency vector is " + str(len(freqData)))
        print("After the filter, the number is " + str(len(filterFreqData)))
        print("")
    
    return newItems

if env.TESTMODE:
    # In the input value, the more frequent value at 50% are 1 and 3.
    assert getListOfRelevantItemInFeatures([[1],[2],[3],[4,3],[5,1],[]],50) == [1, 3]




# @categorizeVectVar : To make [feature vectors] out of categorical values
#--------
# vector: values of the categorical feature (list of list of integers)
# categories: list of the categories (list of integers)
# name: Name of the field corresponding to the values (string)
# log: to print log (boolean)
# @return : the new list with categorical values. The categories are represented with [feature vectors]. (list of list of binary numbers)
#--------
# For further understanding, see example below.
#--------
def categorizeVectVar(vector, categories, name = '', log = False):
    
    wholeNewVector = []
    
    # to indicate the number of items per categories
    countingFreq = np.zeros(len(categories)+1)
    
    for elem in vector:
        
        # A [feature vector] for each item
        newVector = []
        
        # Loop over each categories
        for cat in categories:
            
            if cat in elem :
                # the element contains that category cat
                newVector.append(1)                
            
            else: 
                newVector.append(0)
                
        # Last Category : Unknown, when the element contained no categories listed in the input variable
        newVector.append(0 if 1 in newVector else 1)
        
        # To add each vector of binary numbers gives us the number of items per category.
        countingFreq = np.add(countingFreq, newVector)
        wholeNewVector.append(newVector)
    
    if log:
        print("--------------------------------- " + str(name))
        print("categorizeVectContinuousVar : Vectorizing")       
        print("The categories are :")
        print(categories)
        print("The countingFreq is :")
        print(countingFreq)
        print("The sum of countingFreq is : " + str(sum(countingFreq)))
        print("")
        
    return wholeNewVector

if env.TESTMODE:
    # Each item is now a [feature vector], and the categories are [1, 3, Unknown]
    assert categorizeVectVar([[1],[2],[3],[4,3],[5,1],[]], [1, 3]) == [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]


# @formingColumnsDF : To make the header of columns according to categories in features
#                       (to keep the notion of boudaries)
#--------
# continuousData : list of couple (couple (list): string and list) : Parameter name 
#                   and the inner boundaries used to categorize the values. (example ['runtime', [60,180]])
# categoricalData : list of triple (triple (list): string, list and boolean) : 
#                   Parameter name and the list of categories and boolean True 
#                   if there is an Unknown Category not mentionned in the previous list.
# @return : list of string that the names of the columns
#--------
# Warning : the order is important
# For further understanding, see example below.
#--------
def formingColumnsDF(continuousData, categoricalData):
    columnsDF = []
    
    for name, var in continuousData:
        lastcol = "Min"
        
        for cat in var :
            col = name + "_" + str(lastcol) + "To" + str(cat)
            columnsDF.append(col)
            lastcol = cat
        
        col = name + "_" + str(lastcol) + "ToMax"
        columnsDF.append(col)
        
    for name, var, nullable in categoricalData:
        
        for cat in var :
            columnsDF.append(name + "_" + str(cat))
    
        if nullable:
            columnsDF.append(name + "_" + "unknown")
        
    return columnsDF

if env.TESTMODE:
    # With a Unknown Category for 'genre'
    assert formingColumnsDF([['runtime', [60,180]]], [['genre',['Comedy'], True]]) == ['runtime_MinTo60', 'runtime_60To180', 'runtime_180ToMax', 'genre_Comedy', 'genre_unknown']
    
    # With no Unknown Category for 'genre'
    assert formingColumnsDF([['runtime', [60,180]]], [['genre',['Comedy'], False]]) == ['runtime_MinTo60', 'runtime_60To180', 'runtime_180ToMax', 'genre_Comedy']



# @verifyNbInstancesList : to check if the list of all parameters have the same numbers of instances
#--------
# dataList: data (list of list of any)
# nbinst: number of reference (integer)
# @return: (boolean)
#--------
def verifyNbInstancesList(dataList, nbinst):
    
    for ind in range(len(dataList)):
        
        if len(dataList[ind]) != nbinst:
            return False
        
    return True



# @formingTheDictForDF : To 'transpose' the data and make list by index
#--------
# orderedData: list of list. Each list (a parameter) contains either a simple type or a list.
# @return : list of list. Each list is now an instance.
#--------
# For further understanding, see example below.
#--------
def formingTheDictForDF(orderedData):
    
    nbInstances = len(orderedData[0])
    newData = []
    
    for inst in range(nbInstances):        
        
        # For each instance (a row)
        newInst = []
        
        for var in orderedData:
            
            # For each parameter (a column)
            
            if type(var[inst]) is int or type(var[inst]) is float or type(var[inst]) is str:
                # It is a simple type (int, float, string)
                newInst = newInst + [var[inst]]
            else:
                newInst = newInst + var[inst]
            
        newData.append(newInst)
        
    return newData

if env.TESTMODE:
    assert formingTheDictForDF([[[1],[1],[1]], [[2],[2],[2]], [[3],[3],[3]]]) == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert formingTheDictForDF([[[1,2],[1,2],[1,2]], [3,3,3]]) == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]




# @normalize : To normalize a Numpy.Array by rows
#--------
# df_ : a Numpy.Array
# @return: a numpy.array normalized
#--------
def normalize(df_):
    
    # Empty numpy array
    res = np.zeros(df_.shape)
    
    # To loop over the rows
    for i in range(df_.shape[0]):
        
        # Each row
        row = np.array(df_[i,:])
        
        # Norm is Sum( Sqrt( x * x ) )
        norm_ = np.sqrt((row.dot(row)).sum())
                    
        res[i,:] =  df_[i,:] / norm_
            
    return res

if env.TESTMODE:
    mydf = np.array([[1,2],[3,4]])
    normdf = normalize(mydf)
    assert round(normdf[0,0] * 100) == round(0.4472136 * 100)
    assert round(normdf[0,1] * 100) == round(0.89442719 * 100)
    
    
    
    
# @cosineDistance : To compute the cosine distance between the rows of a Numpy.Array
#--------
# df_ : a  Numpy.Array, the rows are the instances
# @return: a numpy.array of cosine distance between the movies
#--------
def cosineDistance(x):
    return cosine_distances(x, x)

if env.TESTMODE:
    mydf = np.array([[1,2],[3,4]])
    distdf = cosineDistance(mydf)
    assert round(distdf[0,0] * 100) == 0
    assert round(distdf[0,1] * 1000) == round(0.01613009 * 1000)




# @readCsvEvaluationData : To read the evaluation_ratings.csv and get the data
#--------
# evalFile: path to the evaluation_ratings.csv file (string)
# log: to diplay the logs (boolean)
# @return : a Dict that have for key the user_id (int) and for value the list of movie_id (list of strings).
#           (movie_id that should be rated for this user.)
#--------
def readCsvEvaluationData(evalFile, log = False):
    
    evalutionByUser = {}
    
    ifile = open(evalFile, "r", encoding="utf8")
    reader = csv.reader(open(evalFile, "r", encoding="utf8"), delimiter=",")
    
    rownum = 0
    errornum = 0
    
    for row in reader:
        
        try:
                        
            if len(row) < 2:
                errornum += 1
                continue
            
            # to skip the Header
            if row[0] == 'userId':
                continue
            
            
            if rownum % 500000 == 0 and log:
                print("Reading file "+ evalFile +" : line count ... "+ str(rownum))
        
            # user_id 
            user_id = (int(row[0]))
            
            # movie_id
            movie_id = (row[1])        
            
            # Adding the new value to the output dict
            if user_id in evalutionByUser:
                evalutionByUser[user_id].append(movie_id)
            else:
                evalutionByUser[user_id] = [movie_id]
            
            rownum += 1
            
        except:
            
            # Skipped Rows
            errornum += 1
            pass
    
    ifile.close()
    
    if log:
        print("Finish reading file "+ evalFile +" : number of lines : "+ str(rownum))
        print('Number of skipped lines : ' + str(errornum))
    
    return evalutionByUser




# @RMSEeval: To calculate the accuracy of the prediction from files
#--------
# targetFile : path to the file that have the right ratings, the targets (string)
#              The columns in the header are : userId, movieId, rating
# predictionsFile : path to the file that have predictions of the ratings (string)
#              The columns in the header are : user_id, movie_id, ratings
#--------
def RMSEeval(targetFile, predictionsFile):
    
    # Reading Files and Retrieve the column index according to the Headers
    
    t = readcsv(targetFile)
    t_ratingIndex = t[0].index("rating")
    t_movieIndex = t[0].index("movieId")
    t_userIndex = t[0].index("userId")
    
    p = readcsv(predictionsFile)
    p_ratingIndex = p[0].index("ratings")
    p_movieIndex = p[0].index("movie_id")
    p_userIndex = p[0].index("user_id")
    
    targets = {}
    
    # Getting all the target values for the couple (user_id, movie_id)
    for t_elem in t[1:]:
        
        targets[str(t_elem[t_userIndex])+"_"+str(t_elem[t_movieIndex])] = float(t_elem[t_ratingIndex])
    
    somme = 0
    denom = 0
    
    for p_elem in p[1:]:
        
        # Skipping lines that have no rating
        if p_elem[p_ratingIndex] == "":
            continue
        
        keyUserMovie = str(p_elem[p_userIndex])+"_"+str(p_elem[p_movieIndex]) 
        
        # Computing the values the RMSE metric 
        if keyUserMovie in targets:
            somme += ( float(p_elem[p_ratingIndex]) - targets[keyUserMovie] )**2
            denom += 1
            
    if denom == 0:
        return 0
    
    return np.sqrt(somme/denom)

