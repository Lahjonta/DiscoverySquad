import pandas as pd
import numpy as np

# read data from MovieLens 25M Dataset
# movies
movies = pd.read_csv('C:/Users/Janika/Downloads/Uni/Data Science/Semester 4/Data Exploration/ml-25m/movies.csv')
print('Shape of this dataset :', movies.shape)
pd.set_option('display.max_columns', None)
movies.head()

# movie ratings
ratings = pd.read_csv('C:/Users/Janika/Downloads/Uni/Data Science/Semester 4/Data Exploration/ml-25m/ratings.csv')
print('Shape of this dataset :', ratings.shape)
ratings.head()

# movie tags
tags = pd.read_csv('C:/Users/Janika/Downloads/Uni/Data Science/Semester 4/Data Exploration/ml-25m/tags.csv')
print('Shape of this dataset :', tags.shape)
users.head()