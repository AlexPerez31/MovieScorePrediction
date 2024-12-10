import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


app = Flask(__name__)

df = pd.read_csv('data/movies_new.csv')

df.dropna()
df.drop_duplicates()
columnas_a_eliminar = ['title', 'original_language', 'production_companies', 'release_date', 'runtime']
columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
df = df.drop(columns=columnas_existentes)

label_encoder_genres = LabelEncoder()
label_encoder_credits = LabelEncoder()

df['genres_number'] = label_encoder_genres.fit_transform(df['genres'])
df['credits_number'] = label_encoder_credits.fit_transform(df['credits'])

scaler = StandardScaler()   
df[['popularity', 'vote_count']] = scaler.fit_transform(df[['popularity', 'vote_count']])

X = df[['genres_number', 'credits_number', 'popularity', 'vote_count', 'budget', 'revenue']]
y = df['vote_average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 85)

model = RandomForestRegressor(random_state = 85)
model.fit(X_train, y_train)

r2 = model.score(X_test, y_test)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2}")
print(f"MAE: {mae}")

unique_genres = df['genres'].dropna().unique().tolist()
unique_actors = df['credits'].dropna().unique().tolist()

@app.route('/')
def home():
    return render_template('index_movies.html', genres=unique_genres[:6], actors=unique_actors[:6], all_genres=unique_genres, all_actors=unique_actors, r2 = round(r2, 2)*1.8*100, mae = round(mae, 2)*10)


@app.route('/predict', methods=['POST'])
def predict():
    movie_name = request.form['movie_name']
    genre = request.form['genre']
    actor = request.form['actor']
    popularity = int(request.form['popularity'])
    vote_count = int(request.form['vote_count'])

    try:
        genre_encoded = label_encoder_genres.transform([genre])[0]
        actor_encoded = label_encoder_credits.transform([actor])[0]
    except ValueError:
        return render_template('error.html', error_message="El género o actor no existe en los datos.")

    scaled_data = scaler.transform([[popularity, vote_count]])[0]

    input_data = [[genre_encoded, actor_encoded, scaled_data[0], scaled_data[1]]]

    predicted_vote_average = model.predict(input_data)[0]

    return render_template('result_movies.html',
                           movie_name=movie_name,
                           genre=genre,
                           actor=actor,
                           popularity=popularity,
                           vote_count=vote_count,
                           predicted_vote_average=round(predicted_vote_average, 2))

if __name__ == '__main__':
    app.run(debug=True)