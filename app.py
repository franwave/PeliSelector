import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. CONFIGURACIÓN Y CARGA DE DATOS
st.set_page_config(page_title="Antigravity Movie Recs", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Cargar archivos
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Unir datasets
    credits.columns = ['id', 'tittle', 'cast', 'crew']
    df = movies.merge(credits, on='id')
    
    # Función para extraer nombres de las listas (Géneros, Keywords, Cast)
    def get_list(obj):
        if isinstance(obj, str):
            L = []
            for i in ast.literal_eval(obj):
                L.append(i['name'])
            return L
        return []

    # Función para obtener el director
    def get_director(obj):
        if isinstance(obj, str):
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    return i['name']
        return ""

    # Limpieza básica
    df['genres'] = df['genres'].apply(get_list)
    df['keywords'] = df['keywords'].apply(get_list)
    df['cast'] = df['cast'].apply(lambda x: get_list(x)[:3]) # Solo los primeros 3 actores
    df['director'] = df['crew'].apply(get_director)
    
    # Quitar espacios para que "Johnny Depp" sea "JohnnyDepp" (evita confusiones al modelo)
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            return str.lower(x.replace(" ", ""))

    features = ['cast', 'keywords', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].apply(clean_data)

    # Crear la "Sopa de palabras" (Soup)
    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    
    df['soup'] = df.apply(create_soup, axis=1)
    return df

df = load_and_clean_data()

# 2. MODELO DE CIENCIA DE DATOS
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Función de recomendación
def get_recommendations(title):
    try:
        idx = df[df['original_title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:7] # Top 6 películas
        movie_indices = [i[0] for i in sim_scores]
        return df.iloc[movie_indices]
    except:
        return None

# 3. INTERFAZ DE USUARIO (STREAMLIT)
st.title("🚀 Antigravity Movie Recommender")
st.write("Seleccioná una película y te diré cuáles se parecen según su género, director y actores.")

selected_movie = st.selectbox(
    "Escribí el nombre de una película:",
    df['original_title'].values
)

if st.button('Obtener Recomendaciones'):
    recommendations = get_recommendations(selected_movie)
    
    if recommendations is not None:
        st.subheader(f"Si te gustó '{selected_movie}', probá con:")
        
        # Mostrar resultados en columnas
        cols = st.columns(3)
        for i, (index, row) in enumerate(recommendations.iterrows()):
            with cols[i % 3]:
                st.info(f"**{row['original_title']}**")
                st.caption(f"⭐ Rating: {row['vote_average']}")
                with st.expander("Ver resumen"):
                    st.write(row['overview'])
    else:
        st.error("No pudimos encontrar esa película, intentá con otra.")