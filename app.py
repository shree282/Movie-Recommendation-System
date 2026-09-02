import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on genre similarity.")

df = pd.read_csv("movies.csv")
df["genres"] = df["genres"].fillna("").str.replace("|", " ", regex=False)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["genres"])
similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

def recommend(title, n=5):
    idx = indices[title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [x for x in scores if x[0] != idx][:n]
    result = df.iloc[[i for i, _ in scores]][["title", "genres"]].copy()
    result["similarity_score"] = [round(s, 3) for _, s in scores]
    return result

choice = st.selectbox("Select a movie", df["title"].sort_values())
n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend Movies"):
    st.subheader("Recommended Movies")
    st.dataframe(recommend(choice, n), use_container_width=True)

st.caption("Method: TF-IDF vectorization + cosine similarity (content-based filtering).")
