import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load your data (columns: 'text', 'label')
data = pd.read_csv("twitter sentiment analysis.csv")  # Replace with your data

st.title("Twitter Sentiment Dashboard")

# 1. Overall Sentiment Distribution
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots()
data["label"].value_counts().plot(kind="bar", color=["red", "gray", "green"], ax=ax)
st.pyplot(fig)

# 2. Word Clouds for Each Sentiment
st.subheader("Most Frequent Words")

# Positive words
positive_tweets = " ".join(data[data["label"] == "positive"]["text"])
wordcloud_pos = WordCloud(width=600, height=300, background_color="white").generate(positive_tweets)
st.image(wordcloud_pos.to_array(), caption="Positive Words")

# Negative words
negative_tweets = " ".join(data[data["label"] == "negative"]["text"])
wordcloud_neg = WordCloud(width=600, height=300, background_color="black").generate(negative_tweets)
st.image(wordcloud_neg.to_array(), caption="Negative Words")

# 3. Show Sample Tweets
st.subheader("Sample Tweets")
sentiment_filter = st.selectbox("Filter by sentiment:", ["all", "positive", "negative", "neutral"])
if sentiment_filter != "all":
    filtered_tweets = data[data["label"] == sentiment_filter]
else:
    filtered_tweets = data
st.write(filtered_tweets.sample(5))  # Show 5 random tweets

