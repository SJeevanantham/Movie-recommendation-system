# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import difflib
import sklearn
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('C:/Users/Jeeva/Documents/Movie-recommendation-system/movies.csv')
# selecting the relevant features for recommendation
st.title("Movie Recommandation System")
selected_features = ['genres','keywords','tagline','cast','director']

# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
  # combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
# converting the text data to feature vectors

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)
# getting the movie name from the user

movie_name = st.text_input(' Enter your favourite movie name : ')
# creating a list with all the movie names given in the dataset

list_of_all_titles = movies_data['title'].tolist()

# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

if find_close_match:
    close_match = find_close_match[0]
else:
    # Handle the case when the list is empty
    close_match = 'Avatar'  # or any default value you prefer


# finding the index of the movie with title

# Check if close_match exists in the dataset
if close_match in movies_data['title'].values:
    # Retrieve the index of the movie
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
else:
    # Handle the case when close_match doesn't exist in the dataset
    index_of_the_movie = None  # or any default value you prefer
    st.write("Sorry, the movie '{}' was not found in the dataset.".format(close_match))

# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))

# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

# print the name of similar movies based on the index

st.write('Movies suggested for you :')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<6):
    st.write(f"{i}.{title_from_index}")
    i+=1