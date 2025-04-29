import nltk
# Download required NLTK resources at runtime
nltk.download('stopwords')
nltk.download('punkt')
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Comment out these downloads after first time
# nltk.download('stopwords')
# nltk.download('punkt')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = text.split()
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("📧 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    st.write("🔍 Transformed Text:", transformed_sms)

    vector_input = tfidf.transform([transformed_sms])
    proba = model.predict_proba(vector_input)

    if proba[0][1] > 0.4:
        st.error("🚫 This is Spam!")
    else:
        st.success("✅ This is Not Spam!")
