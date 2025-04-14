import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page config
st.set_page_config(
    page_title="LSTM Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cute design
st.markdown("""
<style>
    .main {
        background-color: #FFF5F5;
    }
    .stTextInput>div>div>input {
        background-color: #FFEEEE;
    }
    .stButton>button {
        background-color: #FF9AA2;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #FFB7B2;
        color: white;
    }
    .css-1aumxhk {
        background-color: #FFDAC1;
        background-image: none;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Comic Sans MS', cursive;
    }
    .big-font {
        font-size: 24px !important;
        color: #E27D60;
    }
</style>
""", unsafe_allow_html=True)

# Header with emojis
st.title("😊😠😐 LSTM Sentiment Analysis")
st.markdown("""
<div style="background-color:#FFDAC1;padding:10px;border-radius:10px;margin-bottom:20px;">
    <h3 style="color:#E27D60;text-align:center;">Deep Learning-powered Sentiment Analysis</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header("About This App")
    st.write("""
    This app uses an LSTM neural network to analyze text sentiment.
    It can detect positive 😊, negative 😠, or neutral 😐 emotions in text.
    """)
    st.markdown("---")
    st.write("Powered by:")
    st.write("- LSTM Neural Networks")
    st.write("- Word Embeddings")
    st.write("- Streamlit")
    
    # Add a cute image (replace with your own or remove)
    try:
        image = Image.open("sentiment_image.png")
        st.image(image, caption="Deep Learning Emotions", use_column_width=True)
    except:
        st.info("Add an image named 'sentiment_image.png' to enhance the app")

# Load model and tokenizer
@st.cache_resource
def load_components():
    try:
        # Load the LSTM model
        model = load_model('lstm_sentiment_model.pkl')
        
        # Load the tokenizer (assuming it was saved with the model)
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

model, tokenizer = load_components()

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Function to analyze sentiment
def analyze_sentiment(text, model, tokenizer, max_len=100):
    if model is None or tokenizer is None:
        return "Error", 0.0
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([processed_text])
    
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    
    # Get sentiment and confidence
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    predicted_class = np.argmax(prediction)
    sentiment = sentiment_classes[predicted_class]
    confidence = np.max(prediction)
    
    # Get probabilities for all classes
    probabilities = prediction[0]
    
    return sentiment, confidence, probabilities

# Main app
text_input = st.text_area("Enter your text here:", height=150, 
                         placeholder="Type something like 'I love this product! It's amazing!'")

if st.button("Analyze Sentiment 🚀"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    elif model is None:
        st.error("Model could not be loaded. Please check your model files.")
    else:
        with st.spinner("Analyzing your text with LSTM... 🤖"):
            # Progress bar
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Get prediction
            sentiment, confidence, probabilities = analyze_sentiment(text_input, model, tokenizer)
            
            # Display results
            st.success("LSTM Analysis complete! 🎉")
            
            # Big emoji based on sentiment
            col1, col2, col3 = st.columns(3)
            with col2:
                if sentiment == "Positive":
                    st.markdown("<h1 style='text-align: center;'>😊</h1>", unsafe_allow_html=True)
                elif sentiment == "Negative":
                    st.markdown("<h1 style='text-align: center;'>😠</h1>", unsafe_allow_html=True)
                else:
                    st.markdown("<h1 style='text-align: center;'>😐</h1>", unsafe_allow_html=True)
                
                st.markdown(f"<h2 style='text-align: center; color: #E27D60;'>{sentiment}</h2>", 
                           unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Confidence: {confidence:.0%}</p>", 
                           unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown("### Confidence Level")
            meter_html = f"""
            <div style="background: #FFEEEE; width: 100%; height: 30px; border-radius: 15px;">
                <div style="background: #FF9AA2; width: {confidence*100}%; height: 30px; border-radius: 15px; 
                            text-align: center; color: white; line-height: 30px;">
                    {confidence:.0%}
                </div>
            </div>
            """
            st.markdown(meter_html, unsafe_allow_html=True)
            
            # Sentiment distribution visualization
            st.markdown("### Sentiment Probability Distribution")
            data = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Probability': probabilities
            })
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Sentiment', y='Probability', data=data, 
                        palette=['#FF9AA2', '#FFB7B2', '#FFDAC1'], ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            plt.title('LSTM Sentiment Probabilities')
            st.pyplot(fig)
            
            # Display processed text
            with st.expander("See processed text"):
                processed_text = preprocess_text(text_input)
                st.write("After cleaning and preprocessing:")
                st.code(processed_text)
            
            # Fun facts based on sentiment
            st.markdown("### Did You Know? 💡")
            if sentiment == "Positive":
                st.info("Positive emotions can broaden your mindset and build lasting resources!")
            elif sentiment == "Negative":
                st.info("Acknowledging negative emotions is the first step to addressing them!")
            else:
                st.info("Neutral emotions provide balance and stability in our lives!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #E27D60;'>Thank you for using our LSTM Sentiment Analysis App! ❤️</p>", 
            unsafe_allow_html=True)
