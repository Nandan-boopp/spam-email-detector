import nltk
nltk.download('stopwords')
nltk.download('punkt')
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Stopwords list is now fetched only once for better performance
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = text.split()  # Split text into words

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stop_words and word not in string.punctuation]

    # Apply stemming to the words
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Load pre-trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    st.write("Transformed Text: ", transformed_sms)  # Display transformed text for debugging
    
    # Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict if the message is spam or not
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
