import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained LSTM model
model = pickle.load(open("lstm_sentiment_model.keras", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

st.title("Twitter Sentiment Analysis")

user_input = st.text_area("Enter a tweet:")
if st.button("Analyze"):
    seq = tokenizer.texts_to_sequences([user_input])
    pad_seq = pad_sequences(seq, maxlen=100)
    prediction = model.predict(pad_seq)
    sentiment = ["Negative", "Neutral", "Positive"][np.argmax(prediction)]
    st.write(f"Sentiment: {sentiment}")
