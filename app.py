import streamlit as st
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Spam Mail Detection App")

st.write("Enter a message to check if it is Spam or Not Spam")

user_input = st.text_area("Enter Message")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.Warning("Please enter a message")
    else:
        text_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(text_tfidf)[0]

        if prediction == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")