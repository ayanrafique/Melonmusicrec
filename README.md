# Fantano Recommender

This is a Flask app that recommends music albums based on user input. It uses the MiniLM-L6-v2 model from SentenceTransformers to calculate cosine similarity between the user input and album reviews. The app then returns the top 10 most similar albums.

## Installation

Run the app using `python melonmusicrec.py`

## Usage

1. Enter a sentence or phrase that describes the type of music you want to listen to.
2. Click the "Get Recommendations" button.
3. The app will return the top 10 most similar albums.

## Libraries Used

- pandas
- numpy
- Flask
- SentenceTransformers
- sklearn

## Acknowledgements

This app was inspired by the music reviews of Anthony Fantano, aka The Needle Drop.
