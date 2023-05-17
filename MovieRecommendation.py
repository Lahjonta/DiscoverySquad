import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage import io

# read the dataset https://www.kaggle.com/datasets/akshaypawar7/millions-of-movies
df = pd.read_csv('movies.csv')
pd.set_option('display.max_columns', None)
df.loc[0]['overview']
df.columns
# Test

# preprocess data
# drop movies that have a short description
df['word_count'] = df['overview'].apply(lambda x: len(str(x).split()))
df = df[df['word_count'] >= 20]
df.drop('word_count', axis=1, inplace=True)

# drop all duplicates
df.drop_duplicates(subset=['title', 'release_date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop rows with missing poster_path
df.dropna(subset=['poster_path'], inplace=True)

# fill empty cells
df.fillna(value={i: '' for i in ['overview', 'genres', 'keywords', 'credits']}, inplace=True)

# lambda func for str split join
strOp= lambda x: ' '.join(x.split('-'))

# add keywords, genres and credits to overview for full information
df.overview = df.overview + df.keywords.apply(strOp) + df.genres.apply(strOp) + df.credits.apply(lambda x: ' '.join(x.replace(' ', '').split('-')[:3]))
df.overview[0]

# TF-IDF Vectorizer to transform words into numbers and remove common english words like 'the'
tfidf = TfidfVectorizer(stop_words='english')

# transform data with TF-IDF vectorizer to create matrix
tfidf_matrix = tfidf.fit_transform(df['overview'])

'''
#display some columns with vectorized words
display(pd.DataFrame(
    tfidf_matrix[:5, 10000:10005].toarray(),
    columns= tfidf.get_feature_names_out()[10000:10005],
    index = df.title[:5]).round())

print(tfidf_matrix.shape)
# 974079 different words used to describe movies
'''

def get_recommendation(user_input):
    # Vectorize user input
    vectorized_input = tfidf.transform([user_input])

    # Calculate cosine similarity between user input and movie overviews
    similarity_scores = cosine_similarity(vectorized_input, tfidf_matrix)

    # Get the indices of top similar movies
    top_indices = similarity_scores.argsort()[0][::-1][:3]

    # Retrieve the top recommended movies
    recommended_movies = df.iloc[top_indices]

    # Display movie posters for the recommended movies
    fig, ax = plt.subplots(1, 3, figsize=(15, 20))
    ax = ax.flatten()
    for i, j in enumerate(recommended_movies.poster_path):
        try:
            ax[i].axis('off')
            ax[i].set_title(recommended_movies.iloc[i].title, fontsize=22)
            a = io.imread(f'https://image.tmdb.org/t/p/w500/{j}')
            ax[i].imshow(a)
        except:
            pass
    fig.tight_layout()
    plt.show()

user_input = "Twilight"

get_recommendation(user_input)
    
    


