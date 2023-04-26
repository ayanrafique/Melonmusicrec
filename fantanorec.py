# Import necessary libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the all-MiniLM-L6-v2 model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the merged_reviews_with_embeddings.csv file
data = pd.read_csv('merged_reviews_with_embeddings_spotify.csv')
data['embedding'] = data['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Define function to get recommendations
def get_recommendations(input_text, data):
    input_embedding = model.encode(input_text)
    similarities = cosine_similarity([input_embedding], data['embedding'].tolist())[0]
    data['similarity'] = similarities
    sorted_data = data.sort_values(by=['similarity', 'score', 'title'], ascending=[False, False, True])
    top_10 = sorted_data.head(10)[['title', 'artist', 'score', 'album_cover_url', 'album_url']]
    top_10['embed_url'] = top_10['album_url'].apply(lambda x: x.replace('https://open.spotify.com/album/', 'https://open.spotify.com/embed/album/') if isinstance(x, str) else None)
    return top_10[['title', 'artist', 'score', 'album_cover_url', 'embed_url']]

# Replace index route
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    if request.method == 'POST':
        input_text = request.form['user_input']
        recommendations = get_recommendations(input_text, data).to_dict(orient='records')
    return render_template('home.html', recommendations=recommendations)

# Remove the recommendations route

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
