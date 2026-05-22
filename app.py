import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 نظام ترشيح الأفلام")


@st.cache_resource
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')
    return movies, ratings

movies, ratings = load_data()


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
svd = TruncatedSVD(n_components=12, random_state=42)
matrix_reduced = svd.fit_transform(csr_matrix(user_movie_matrix.values))
predicted_ratings = np.dot(matrix_reduced, svd.components_)
predicted_df = pd.DataFrame(predicted_ratings, columns=user_movie_matrix.columns, index=user_movie_matrix.index)

def get_hybrid_rec(user_id, title, top_n=5):
    if title not in movies['title'].values: return []
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]

    results = []
    for i, score in sim_scores:
        m_id = movies.iloc[i]['movieId']
        m_title = movies.iloc[i]['title']

        collab_score = predicted_df.loc[user_id, m_id] if user_id in predicted_df.index and m_id in predicted_df.columns else 0
        final_score = (0.5 * collab_score) + (0.5 * score)
        results.append((m_title, movies.iloc[i]['genres'], final_score))

    return sorted(results, key=lambda x: x[2], reverse=True)[:top_n]


col1, col2 = st.columns(2)
with col1:
    u_id = st.number_input("أدخل معرف المستخدم (User ID):", min_value=1, value=1)
with col2:
    m_name = st.selectbox("اختر فيلماً أعجبك:", movies['title'].values)

if st.button("عرض الترشيحات"):
    recs = get_hybrid_rec(u_id, m_name)
    for t, g, s in recs:
        st.success(f"**{t}**")
        st.caption(f"التصنيف: {g} | درجة التوافق: {s:.2f}")