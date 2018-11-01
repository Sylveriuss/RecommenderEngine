Recommender Engine
By Surya Sylverius
------------------

Goal:

The purpose of this project is to build a Recommender system that will predict user's movies ratings.
We have the knowledge of previous ratings and the movies' metadata.


Approach:

This is a content-based recommendation system. 
We study the content of each movie. The metadata of the movies provide a vector of features like the 'release date', the 'genre'.
These are used to caracterize each movie. 
We determine the user's profile from the movies that he has rated.
The profile have the same vector of features as the movies.
We can then use this user's profile to predict the rating he could give to another movie (a dot product between the two vectors).

In machine learning, this can be modelized as a linear regression problem.
Given a set of movies (x) and their ratings (t), can we find a profile (U) that predict these ratings ?
To solve this, I decided to make a gradient descent (over a number of epochs and with learning rate).

My first approach was to build TF-IDF model and use it to predict ratings. (see tfIdf_example.py)
The best result have been from the linear regression gradient descent method.


Data:

movies_metadata.csv : 
	adult,belongs_to_collection,budget,genres,homepage,id,imdb_id,original_language,original_title,overview,popularity,poster_path,production_companies,production_countries,release_date,revenue,runtime,spoken_languages,status,tagline,title,video,vote_average,vote_count

ratings.csv :
	user_id,movie_id,rating,timestamp
	
evaluation_ratings.csv:
	user_id,movie_id


Environnement:

The code is in python. Some librairies have been used : Numpy, Scipy, Scikit-learn, Pandas


	
Run:

This files contain my work to find to best solution. 
To run the application to make prediction for the evaluation_ratings file, the Main.py file has the principal function.

In a terminal console, run Main.py with the arguments as following :

> python Main.py [$OutputFilePath] [$DirectoryOfInputDataFiles] [(optional)-v]

OutputFilePath : a string
DirectoryOfInputDataFiles : a string. If it is the current directory, put ./
Log : write -v if you want logs.


Warning:

Computation might take some time since the work on parallelization have not been done.