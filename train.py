import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download NLTK resources if not already done
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Load your CSV
df = pd.read_csv("emails.csv")

# Apply the same transform to training data
df['transformed_text'] = df['text'].apply(transform_text)

# Features and labels
X = df['transformed_text']
y = df['spam']  # use the spam column directly

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Training done, model and vectorizer saved.")
