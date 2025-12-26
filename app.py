import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="UBA Data Science - Movie Recs", layout="wide")

# --- 2. FUNCIONES DE APOYO (API Y LIMPIEZA) ---
def fetch_poster(movie_id):
    # REEMPLAZÁ ESTO CON TU KEY DE TMDB
    api_key = st.secrets["TMDB_API_KEY"]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=es-ES"
    try:
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return None
    return "https://via.placeholder.com/500x750?text=Sin+Poster"

def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ""

def get_list(obj):
    if isinstance(obj, str):
        return [i['name'] for i in ast.literal_eval(obj)]
    return []

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    return str.lower(x.replace(" ", ""))

# --- 3. CARGA Y PROCESAMIENTO DE DATOS ---
@st.cache_data
def load_and_clean_data():
    # Carga
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Merge (Aquí se crea 'df')
    credits.columns = ['id', 'tittle', 'cast', 'crew']
    df = movies.merge(credits, on='id')
    
    # --- Weighted Rating (IMDb Formula) ---
    # $$WR = (\frac{v}{v+m} \cdot R) + (\frac{m}{v+m} \cdot C)$$
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.9)
    
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)
    
    df['score'] = df.apply(weighted_rating, axis=1)
    
    # Año de lanzamiento
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    
    # Limpieza para el modelo
    df['genres'] = df['genres'].apply(get_list)
    df['keywords'] = df['keywords'].apply(get_list)
    df['cast'] = df['cast'].apply(lambda x: get_list(x)[:3])
    df['director'] = df['crew'].apply(get_director)
    
    features = ['cast', 'keywords', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].apply(clean_data)
        
    df['soup'] = df.apply(lambda x: ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']), axis=1)
    return df

df = load_and_clean_data()

# --- 4. CÁLCULO DE SIMILITUD (CIENCIA DE DATOS) ---
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

def get_recommendations(title, year_min):
    if title not in df['original_title'].values:
        return None
    
    idx = df[df['original_title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Tomamos 30 candidatos para filtrar por año después
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    
    similar_movies = df.iloc[movie_indices].copy()
    
    # Filtro por año del sidebar
    similar_movies = similar_movies[similar_movies['year'] >= year_min]
    
    # Ordenamos por score (calidad estadística)
    return similar_movies.sort_values('score', ascending=False).head(6)

# --- 5. INTERFAZ (STREAMLIT) ---
st.title("🎬 Antigravity Movie Recommender")
st.markdown("Sistema Híbrido: Similitud de Contenido + Calificación Ponderada de IMDb")

# Sidebar
st.sidebar.header("Filtros Inteligentes")
year_input = st.sidebar.slider("Año mínimo de estreno", 1970, 2017, 2000)

# Selector Principal
movie_list = df['original_title'].values
selected_movie = st.selectbox("Elegí una película que te haya gustado:", movie_list)

if st.button('Obtener Recomendaciones'):
    res = get_recommendations(selected_movie, year_input)
    
    if res is not None and not res.empty:
        st.subheader(f"Basado en tu interés por {selected_movie}:")
        
        # Grid de resultados
        cols = st.columns(3)
        for i, (index, row) in enumerate(res.iterrows()):
            with cols[i % 3]:
                poster = fetch_poster(row['id'])
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"**{row['original_title']}** ({row['year']})")
                st.caption(f"⭐ Calidad: {round(row['score'], 2)}")
                with st.expander("Ver Resumen"):
                    st.write(row['overview'])
    else:
        st.warning("No se encontraron películas que coincidan con los filtros. Probá bajando el año mínimo.")