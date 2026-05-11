import streamlit as st
import pickle
import numpy as np
import os

# Load precomputed files using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
similarity = pickle.load(open(os.path.join(BASE_DIR, 'similarity.pkl'), 'rb'))
pt = pickle.load(open(os.path.join(BASE_DIR, 'pt.pkl'), 'rb'))
books_df = pickle.load(open(os.path.join(BASE_DIR, 'books.pkl'), 'rb'))

st.title("📚 Book Recommendation System")
user_input = st.text_input("Enter a book name:")

def recommend(book_name):
    if book_name not in pt.index:
        return []
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []
    for i in similar_items:
        title = pt.index[i[0]]
        temp_df = books_df[books_df['Book-Title'] == title]
        if len(temp_df) == 0:
            continue
        data.append({
            "title": temp_df['Book-Title'].values[0],
            "author": temp_df['Book-Author'].values[0],
            "image": temp_df['Image-URL-M'].values[0]
        })
    return data

if st.button("Recommend"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid book name.")
    else:
        recommendations = recommend(user_input.strip())
        if len(recommendations) == 0:
            st.error("❌ Book not found! Try another name.")
            st.info("Try one of these: 1984, Animal Farm, Angels & Demons, A Walk to Remember, About a Boy")
        else:
            st.subheader("Top 5 Recommended Books:")
            cols = st.columns(len(recommendations))
            for idx, book in enumerate(recommendations):
                with cols[idx]:
                    try:
                        st.image(book["image"], use_column_width=True)
                    except Exception:
                        st.write("🖼️ No image")
                    st.write(f"**{book['title']}**")
                    st.caption(book["author"])
