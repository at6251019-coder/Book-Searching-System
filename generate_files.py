import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import os
BASE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE, 'Csv files')

# Load your CSV files
books = pd.read_csv(os.path.join(CSV_DIR, 'Books.csv.zip'))
ratings = pd.read_csv(os.path.join(CSV_DIR, 'Ratings.csv.zip'))
users = pd.read_csv(os.path.join(CSV_DIR, 'Users.csv.zip'))

# Merge the data
ratings_with_name = ratings.merge(books, on='ISBN')

# Filter: only popular books (adjust numbers as needed)
x = ratings_with_name.groupby('Book-Title').count()['Book-Rating'] > 50
famous_books = x[x].index
final_ratings = ratings_with_name[ratings_with_name['Book-Title'].isin(famous_books)]

# Remove duplicate user-book ratings
final_ratings = final_ratings.drop_duplicates(['User-ID', 'Book-Title'])

# Create pivot table
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)

# Compute similarity matrix
similarity = cosine_similarity(pt)

# Save files
pickle.dump(books, open(os.path.join(BASE, 'books.pkl'), 'wb'))
pickle.dump(pt, open(os.path.join(BASE, 'pt.pkl'), 'wb'))
pickle.dump(similarity, open(os.path.join(BASE, 'similarity.pkl'), 'wb'))

print("✅ All files generated successfully! (books.pkl, pt.pkl, similarity.pkl)")
