import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load data ----
base_path = r'C:\Users\Ansh Thakur\OneDrive\Documents\AI PROJECT\Csv files'

books = pd.read_csv(os.path.join(base_path, 'Books.csv.zip'), low_memory=False)
users = pd.read_csv(os.path.join(base_path, 'Users.csv.zip'), low_memory=False)
ratings = pd.read_csv(os.path.join(base_path, 'Ratings.csv.zip'), low_memory=False)

# ---- Preprocessing ----
ratings_with_name = ratings.merge(books, on='ISBN')

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
    ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']
]

x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# ---- Similarity matrix ----
similarity_scores = cosine_similarity(pt)

# ---- Function for Flask ----
def recommend_books(book_name, top_n=5):
    """
    Recommend top N similar books for a given book title.
    Returns list of dictionaries: [{title, author, image}, ...]
    """
    if book_name not in pt.index:
        return []

    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        for _, row in temp_df.iterrows():
            data.append({
                'title': row['Book-Title'],
                'author': row['Book-Author'],
                'image': row['Image-URL-M']
            })
    return data


import pickle

# Save precomputed files
with open('pt.pkl', 'wb') as f:
    pickle.dump(pt, f)

with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity_scores, f)

with open('books.pkl', 'wb') as f:
    pickle.dump(books, f)

print("Pickle files created successfully!")
