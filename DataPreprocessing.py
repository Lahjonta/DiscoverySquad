import pandas as pd
import numpy as np

# Read data
## Movie titles and genres
movies = pd.read_csv('ml-25m/movies.csv')
print('Shape of this dataset :', movies.shape)
pd.set_option('display.max_columns', None)
movies.head()

## movie ratings
ratings = pd.read_csv('ml-25m/ratings.csv')
print('Shape of this dataset :', ratings.shape)
ratings.head()

## tags
tags = pd.read_csv('ml-25m/tags.csv')
print('Shape of this dataset :', tags.shape)
tags.head()

# Data Preprocessing
rate = ratings.copy()
tag = tags.copy()
titles = movies.copy()
del rate['timestamp']
del tag['timestamp']

## Separate the Years from titles
year = titles['title'].str.findall('\((\d{4})\)').str.get(0)
titles['Year'] = year

## Separate the genres in different columns and count genres
genres = titles['genres'].str.split(pat='|', expand=True).fillna(0)
genres.columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10']
cols = genres.columns
genres[cols] = genres[cols].astype('category')
genres1 = genres.copy()
cat_columns = genres1.select_dtypes(['category']).columns

#count genres(no zeros)
genres1[cat_columns] = genres1[cat_columns].apply(lambda x: x.cat.codes)
genres1['genre_count'] = genres1[cols].gt(0).sum(axis=1) #count > 0

#assign to dataframe
titles['genre_count'] = genres1['genre_count']
titles[cols] = genres[cols]

titles.head()

## Calculate average movie rating 
rating_avg = rate.groupby('movieId')['rating'].mean().reset_index()
rating_avg = pd.DataFrame(rating_avg)

rating_count = rate.groupby('movieId')['rating'].count().reset_index()
rating_count = pd.DataFrame(rating_count)
rating_count.rename({'rating': 'rating_count'}, axis=1, inplace=True)

movie_rating = rating_avg.merge(rating_count, on='movieId', how='inner')


## Collect the tags left by users for each movie
### Load the tags data

### Group the merged data by movieId and tag and count the number of occurrences
grouped_tags = tags.groupby(['movieId', 'tag']).size().reset_index(name='count')

### Filter out tags that occur less than 5 times
grouped_tags = grouped_tags[grouped_tags['count'] >= 100]

### Pivot the grouped data to create a matrix of movies and tags
pivoted_df = grouped_tags.pivot_table(index='movieId', columns='tag', values='count', fill_value=0)

### Rename the columns to remove spaces and convert to lowercase
pivoted_df.columns = [col.lower().replace(' ', '_') for col in pivoted_df.columns]


## Merge average ratings and tags with titles dataframe to create main table
titles = titles.merge(movie_rating, on='movieId', how='inner')
titles.rename({'rating': 'avg_rating'}, axis=1, inplace=True)
titles = titles.merge(pivoted_df, on='movieId', how='inner')
titles.head(3)

## Save main table as .csv
