# -*- coding: utf-8 -*-
# Environnement Variables used throught the code

# @TESTMODE : if True, assertions will test the functions.
TESTMODE = True

LEARNINGRATE = 0.5
EPOCHS = 5

FEATURE_TYPE = "INTERMEDIATE"
RATING_TYPE = "DOTPRODUCT"


# Input data files

IN_RATINGS="ratings.csv"
IN_MOVIES_METADATA="movies_metadata.csv"
IN_EVALUATION_RATINGS="evaluation_ratings.csv"


# Movie Metadata Processing output filenames

MMDT_DATAFRAME="movieDF.dat"
MMDT_ROWINDEX="movieIndex.dat"
MMDT_COLUMNS="moviesColumns.csv"
