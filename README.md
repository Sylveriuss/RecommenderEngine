# Recommender Engine By Surya Sylverius



### Goal:

The purpose of this project is to build a Recommender system that will predict user's movies ratings.
We have the knowledge of previous ratings and the movies' metadata.




### Approach:

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




### Data:

- movies_metadata.csv : Contain the metadata of movies

- ratings.csv : Movie's ratings by users

- evaluation_ratings.csv : Couple of User and Movie, whose ratings should be predicted




### Environnement:

The code is in python. Some librairies have been used : Numpy, Scipy, Scikit-learn, Pandas.

The files : 
- Env.py : Some global variables that are shared among those files.
- Main.py : Has the main function Main() that we will read all input files, train and predict. It also has a function Evaluate() to train, predict and validate on a test data with the RMSE metric.
- LRPredictor.py : Has the body of the training and prediction part.
- MovieMetadataReadear.py : To read the movie metadata file and process it. The output is saved  in a .dat file, to avoid doing it repeatedly. It provides functions to retrieve data from those .dat files and to clean them.
- LinearRegressionGradientDescent.py : It has the training function that is will compute the gradient descent.
- Tools.py : Directory of functions needed for this implementation

Other :
- MovieDistanceComputer.py : For the generated .dat file, from movies' metadata processing, to compute the distance matrix between the movies (with the cosine distance). The application takes an important amount of time.
- tfIdf_example.py : It contains the functions to compute the TF-IDF algorithm. We can use them on Pandas DataFrame.



### Run:

This files contain my work to find to best solution. 
To run the application to make prediction for the evaluation_ratings file, the Main.py file has the principal function.

In a shell, run Main.py with the arguments as following :

```shell
> python Main.py [$OutputFilePath] [$DirectoryOfInputDataFiles] [(optional)-v]
```

- OutputFilePath : a string
- DirectoryOfInputDataFiles : a string. If it is the current directory, put ./
- Log : write -v if you want logs.


### Warning:

Computation might take some time since work on parallelization have not been done.
