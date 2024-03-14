from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv('imdb_top_1000.csv')
df['Genre'].fillna('', inplace=True)
df['Overview'].fillna('', inplace=True)
df = df.dropna()
df['Genre'] = df['Genre'].apply(lambda x: ' '.join(x.split(',')))
df['combined_features'] = df['Genre'] + ' ' + df['Overview'] +  ' ' + df['Director']

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['Series_Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['Series_Title'].iloc[movie_indices].values.tolist()

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.form['message']
    
    # Recommendation logic goes here
    if data.lower() in ['hi', 'hello', 'hey']:
        response = "Hello! How can I assist you today?"
    elif data.lower() in ['bye', 'goodbye']:
        response = "Goodbye! Have a great day!"
    else:
        try:
            recommendations = get_recommendations(data)
            response = "Here are some movie recommendations for you:\n" + "<br>".join(recommendations)
        except IndexError:
            response = "Sorry, I couldn't find any recommendations for that movie. Please try another one."
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
