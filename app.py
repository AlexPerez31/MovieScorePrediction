import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el dataset y preprocesarlo
df = pd.read_csv('data/movies_new.csv')

# Preprocesamiento de datos
df.dropna()
df.drop_duplicates()

columnas_a_eliminar = ['title', 'original_language', 'production_companies', 'release_date', 'runtime']
columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
df = df.drop(columns=columnas_existentes)

# Codificación de variables categóricas
label_encoder_genres = LabelEncoder()
label_encoder_credits = LabelEncoder()

df['genres_number'] = label_encoder_genres.fit_transform(df['genres'])
df['credits_number'] = label_encoder_credits.fit_transform(df['credits'])

# Escalar características numéricas
scaler = StandardScaler()
df[['popularity', 'vote_count']] = scaler.fit_transform(df[['popularity', 'vote_count']])

# Separar las características y el objetivo
X = df[['genres_number', 'credits_number', 'popularity', 'vote_count']]
y = df['vote_average']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Calcular el coeficiente de determinación (R²)
r2_score = model.score(X_test, y_test)

unique_genres = df['genres'].dropna().unique().tolist()
unique_actors = df['credits'].dropna().unique().tolist()

@app.route('/')
def home():
    return render_template('index_movies.html', genres=unique_genres[:6], actors=unique_actors[:6], all_genres=unique_genres, all_actors=unique_actors)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    movie_name = request.form['movie_name']
    genre = request.form['genre']
    actor = request.form['actor']
    popularity = int(request.form['popularity'])
    vote_count = int(request.form['vote_count'])

    # Preprocesar los datos de entrada
    try:
        genre_encoded = label_encoder_genres.transform([genre])[0]
        actor_encoded = label_encoder_credits.transform([actor])[0]
    except ValueError:
        return render_template('error.html', error_message="El género o actor no existe en los datos.")

    # Escalar las características numéricas
    scaled_data = scaler.transform([[popularity, vote_count]])[0]

    # Crear el vector de entrada para el modelo
    input_data = [[genre_encoded, actor_encoded, scaled_data[0], scaled_data[1]]]

    # Realizar la predicción
    predicted_vote_average = model.predict(input_data)[0]

    # Mostrar los resultados
    return render_template('result_movies.html',
                           movie_name=movie_name,
                           genre=genre,
                           actor=actor,
                           popularity=popularity,
                           vote_count=vote_count,
                           predicted_vote_average=round(predicted_vote_average, 2),
                           model_accuracy=round(r2_score, 2))

if __name__ == '__main__':
    app.run(debug=True)
