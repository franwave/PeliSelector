# 🎬 Antigravity Movie Recommender (Peliselecter)

Una aplicación web interactiva para obtener recomendaciones de películas basada en contenido, construida con **Python** y **Streamlit**.

## 📋 Descripción

**Peliselecter** es un sistema de recomendación que sugiere películas similares a una seleccionada por el usuario. Utiliza técnicas de procesamiento de lenguaje natural y aprendizaje automático para analizar el contenido de las películas (género, director, elenco y palabras clave) y encontrar las coincidencias más cercanas.

## ✨ Características

*   **Recomendaciones Inteligentes**: Sugerencias basadas en similitud de contenido (Cosine Similarity).
*   **Análisis Multidimensional**: Considera múltiples factores como géneros, palabras clave, director y los actores principales.
*   **Interfaz Simple e Intuitiva**: Selecciona una película de la lista y obtén resultados al instante.
*   **Detalles de Películas**: Visualiza el título original, calificación y un resumen de la trama de cada recomendación.

## 🛠️ Tecnologías Utilizadas

*   **[Python](https://www.python.org/)**: Lenguaje principal.
*   **[Streamlit](https://streamlit.io/)**: Framework para la interfaz web.
*   **[Pandas](https://pandas.pydata.org/)**: Manipulación y análisis de datos.
*   **[Scikit-learn](https://scikit-learn.org/)**: Algoritmos de Machine Learning (CountVectorizer, Cosine Similarity).

## 🚀 Instalación y Uso

Sigue estos pasos para ejecutar la aplicación en tu entorno local:

1.  **Clona o descarga el repositorio** en tu carpeta local.

2.  **Instala las dependencias necesarias**:
    Asegúrate de tener Python instalado y ejecuta:
    ```bash
    pip install streamlit pandas scikit-learn
    ```

3.  **Asegúrate de tener los datos**:
    Verifica que los archivos `tmdb_5000_credits.csv` y `tmdb_5000_movies.csv` estén en el mismo directorio que `app.py`.

4.  **Ejecuta la aplicación**:
    En tu terminal, navega al directorio del proyecto y corre el siguiente comando:
    ```bash
    streamlit run app.py
    ```

5.  **¡Listo!** La aplicación se abrirá en tu navegador predeterminado (usualmente en http://localhost:8501).

## 📂 Estructura del Proyecto

*   `app.py`: Código principal de la aplicación.
*   `tmdb_5000_movies.csv`: Dataset con información de las películas.
*   `tmdb_5000_credits.csv`: Dataset con información de créditos (elenco, equipo).
*   `README.md`: Este documento.
