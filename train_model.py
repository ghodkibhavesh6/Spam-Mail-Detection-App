# train_model.py  (FIXED)

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#load dataset
df = pd.read_csv("Spam data.csv")

#use correct columns
df = df[['label', 'text']]
df.columns = ['label', 'message']

#convert label into number
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#remove missing value
df = df.dropna()

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

#TFIDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#train
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

#predict
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved")