# -*- coding: utf-8 -*-
#
# Gradient Descent Computation for Linear Regression 
#

import env
import numpy as np
import Tools as rtools


# @LinearRegressionGradientDescent : Compute the gradient descent for LinearRegression
#-------
# userRatedMovies : list of numpy.array and float (example [ np.array([1,2,0,1,2]), 5, np.array([1,2,7,1,2]), 0])
#                   Alternatively, there will be each instance and their ratings (x and t).
#                   In the example : x_1 = np.array([1,2,0,1,2]), t_1 = 5, x_2 = np.array([1,2,7,1,2]), t_1 = 0
# @return : a numpy.array (same as x) but will be the result of the gradient descent.
#           It will be the user profile from these ratings.
#-------
def LinearRegressionGradientDescent(userRatedMovies, epochs = 0, learningRate = 0):

    if len(userRatedMovies) == 0:
        return
    
    
    # Number of Epochs to learn for one regression.
    if epochs == 0:
        epochs = env.EPOCHS
    
    # Learning Rate for the gradient Descent
    if learningRate == 0:
        learningRate = env.LEARNINGRATE
        
        
    # The User Profile Initialization : The first movie's profile
    theta = np.transpose(np.array(userRatedMovies[0]))
    
    # Number of movies 
    numMovies = int(len(userRatedMovies) / 2)
    
    # For every epoch
    for epoch in range(epochs):
        
        # For every movie
        for movie_i in range(numMovies):
            
            # To get the index in input list
            num_i = movie_i * 2
            
            # Movie's profile : X
            xi = userRatedMovies[num_i]
            
            # Rating : T
            ti = userRatedMovies[num_i + 1]
            
            # Computing Y
            yi = theta * xi
            
            # Calculate the loss Y-T
            loss = yi - ti
            
            # Update the User Profile - LearningRate * loss * X
            theta = theta - learningRate * loss * np.transpose(xi)
    
    return theta

if env.TESTMODE:
    
    x1 = rtools.normalize(np.array([[1,0,0,1,0]]))[0]
    t1 = 5
    x2 = rtools.normalize(np.array([[0,1,1,0,0]]))[0]
    t2 = 0
    
    test = [x1, t1, x2, t2]
    
    userProfile = LinearRegressionGradientDescent(test, 1, 0.8)
    
    y1 = round( x1.dot(userProfile) * 2) / 2
    y2 = round( x2.dot(userProfile) * 2) / 2
    
    assert abs(y1 - t1) <=  1 
    assert abs(y2 - t2) <=  1 