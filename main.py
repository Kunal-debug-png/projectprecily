from flask import Flask, render_template, request
import csv
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load the spaCy model
nlp = spacy.load('en_core_web_md')

# Define a function to calculate semantic similarity
def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    embedding1 = doc1.vector.reshape(1, -1)
    embedding2 = doc2.vector.reshape(1, -1)
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    # Get the input sentences from the HTML form
    text1 = request.form['text1']
    text2 = request.form['text2']

    # Calculate the similarity score
    similarity_score = calculate_similarity(text1, text2)

    # Determine the similarity label
    similarity_label = 1 if similarity_score >= 0.5 else 0

    return render_template('result.html', similarity_label=similarity_label)

if __name__ == '__main__':
    app.run()
