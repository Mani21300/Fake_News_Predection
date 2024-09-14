import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import nltk

# Download NLTK resources if not already downloaded
nltk.download('stopwords')

# Load the trained model and TF-IDF vectorizer
with open("naive_bayes_classifier.pkl", "rb") as f:
    naive_bayes_classifier = pickle.load(f)
with open("tf_idf.pkl", "rb") as f:
    tf_idf = pickle.load(f)

# Preprocessing function
def preprocessing(text):
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    # Stem words
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text

def predict_news_type(news):
    if not news.strip():  # Handle empty input
        return "Please enter some text."

    input_data = {"text": [news]}
    new_def_test = pd.DataFrame(input_data)
    new_def_test["text"] = new_def_test["text"].apply(preprocessing)
    vectorized_input_data = tf_idf.transform(new_def_test["text"])  # Ensure transformation is on the column, not DataFrame
    prediction = naive_bayes_classifier.predict(vectorized_input_data)

    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Real News"

def main():
    st.title("Fake News Detection")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Fake News Detection App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create an input field for the news text
    news_text = st.text_area("Enter News Text")

    # Make prediction when the "Predict" button is clicked
    if st.button("Predict"):
        result = predict_news_type(news_text)
        st.success(f'The prediction is: {result}')

if __name__ == '__main__':
    main()
