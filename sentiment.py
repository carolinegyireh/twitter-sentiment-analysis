import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import re
from PIL import Image

try:
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
except ImportError as e:
    st.error(f"Required dependencies not found: {str(e)}")
    st.stop()

# Page config for Twitter theme
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme
st.markdown("""
<style>
    .main {
        background-color: #E8F5FE;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        border: 1px solid #1DA1F2;
    }
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1991DA;
        color: white;
    }
    .css-1aumxhk {
        background-color: #FFFFFF;
        background-image: none;
    }
    .positive {
        background-color: #E1F9F1;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #17BF63;
    }
    .negative {
        background-color: #FCE8E6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #E0245E;
    }
    .neutral {
        background-color: #F7F9FA;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #657786;
    }
    .twitter-font {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Header with Twitter theme
st.title("🐦 Twitter Sentiment Analysis")
st.markdown("""
<div style="background-color:#1DA1F2;padding:10px;border-radius:10px;margin-bottom:20px;">
    <h3 style="color:white;text-align:center;">Analyze tweets with LSTM deep learning</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar with Twitter info
with st.sidebar:
    st.header("Twitter Analysis")
    st.write("""
    This app specializes in analyzing tweet sentiment using an LSTM model.
    It handles Twitter-specific elements like hashtags and mentions.
    """)
    st.markdown("---")
    st.write("**Twitter Features Supported:**")
    st.write("- Hashtags (#)")
    st.write("- Mentions (@)")
    st.write("- Short forms & slang")
    
    st.markdown("---")
    st.write("**Model Info:**")
    st.write("- LSTM Neural Network")
    st.write("- Trained on Twitter data")
    st.write("- Optimized for social media text")

# Load model and tokenizer
@st.cache_resource
def load_components():
    try:
        # Load the LSTM model
        model = load_model('lstm_sentiment_model.keras')
        
        # Load the tokenizer
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

model, tokenizer = load_components()

# Twitter text preprocessing
def preprocess_tweet(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\'.,!?;:]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Function to analyze sentiment
def analyze_tweet_sentiment(text, model, tokenizer, max_len=50):
    if model is None or tokenizer is None:
        return "Error", 0.0, None, text
    
    # Preprocess the tweet
    processed_text = preprocess_tweet(text)
    
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
    
    return sentiment, confidence, probabilities, processed_text

# Main app
text_input = st.text_area("Enter a tweet or Twitter text:", height=100, 
                         placeholder="Paste a tweet here...", key="tweet_input")

if st.button("Analyze Tweet 🚀"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    elif model is None:
        st.error("Model could not be loaded. Please check your model files.")
    else:
        with st.spinner("Analyzing tweet with LSTM... 🤖"):
            # Progress bar
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Get prediction
            sentiment, confidence, probabilities, processed_text = analyze_tweet_sentiment(text_input, model, tokenizer)
            
            # Display results
            st.success("Tweet analysis complete! 🎉")
            
            # Columns for layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Sentiment icon and confidence
                st.markdown("### Sentiment")
                if sentiment == "Positive":
                    st.markdown("<h1 style='text-align: center; color: #17BF63;'>😊</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div class='positive'><h3 style='color: #17BF63;'>Positive</h3><p>Confidence: {confidence:.0%}</p></div>", unsafe_allow_html=True)
                elif sentiment == "Negative":
                    st.markdown("<h1 style='text-align: center; color: #E0245E;'>😠</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div class='negative'><h3 style='color: #E0245E;'>Negative</h3><p>Confidence: {confidence:.0%}</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<h1 style='text-align: center; color: #657786;'>😐</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div class='neutral'><h3 style='color: #657786;'>Neutral</h3><p>Confidence: {confidence:.0%}</p></div>", unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown("### Confidence Level")
                meter_color = "#17BF63" if sentiment == "Positive" else "#E0245E" if sentiment == "Negative" else "#657786"
                meter_html = f"""
                <div style="background: #E1E8ED; width: 100%; height: 20px; border-radius: 10px;">
                    <div style="background: {meter_color}; width: {confidence*100}%; height: 20px; border-radius: 10px;">
                    </div>
                </div>
                <p style="text-align: right; margin-top: 5px;">{confidence:.0%}</p>
                """
                st.markdown(meter_html, unsafe_allow_html=True)
            
            with col2:
                # Sentiment probabilities chart
                st.markdown("### Sentiment Probabilities")
                data = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Probability': probabilities
                })
                
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = ['#E0245E', '#657786', '#17BF63']
                ax.bar(data['Sentiment'], data['Probability'], color=colors)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Tweet Sentiment Probabilities', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Display processed text
            with st.expander("View processed tweet text"):
                st.write("**Original tweet:**")
                st.write(text_input)
                st.write("**After preprocessing:**")
                st.code(processed_text)
            
            # Twitter insights
            st.markdown("### Twitter Insights 🧠")
            if sentiment == "Positive":
                st.info("""
                **Positive tweets** like this often:
                - Contain uplifting or happy content
                - May have exclamation points for emphasis
                - Often include positive words
                """)
            elif sentiment == "Negative":
                st.info("""
                **Negative tweets** like this often:
                - Express complaints or frustrations
                - Sometimes use ALL CAPS for emphasis
                - Often include negative words
                """)
            else:
                st.info("""
                **Neutral tweets** like this often:
                - Share factual information or news
                - May be objective statements
                - Frequently lack strong emotional language
                """)

# Footer with Twitter theme
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #1DA1F2;">
    <p style="font-weight: bold;">Twitter Sentiment Analysis App</p>
    <p>Analyze the emotions behind tweets with AI</p>
    <p>Author: C.Gyireh</p>
</div>
""", unsafe_allow_html=True)import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import re
from PIL import Image

try:
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
except ImportError as e:
    st.error(f"Required dependencies not found: {str(e)}")
    st.stop()

# Page config for Twitter theme
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme
st.markdown("""
<style>
    .main {
        background-color: #E8F5FE;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        border: 1px solid #1DA1F2;
    }
    .stButton>button {
        background-color: #1DA1F2;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1991DA;
        color: white;
    }
    .css-1aumxhk {
        background-color: #FFFFFF;
        background-image: none;
    }
    .positive {
        background-color: #E1F9F1;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #17BF63;
    }
    .negative {
        background-color: #FCE8E6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #E0245E;
    }
    .neutral {
        background-color: #F7F9FA;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #657786;
    }
    .twitter-font {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Header with Twitter theme
st.title("🐦 Twitter Sentiment Analysis")
st.markdown("""
<div style="background-color:#1DA1F2;padding:10px;border-radius:10px;margin-bottom:20px;">
    <h3 style="color:white;text-align:center;">Analyze tweets with LSTM deep learning</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar with Twitter info
with st.sidebar:
    st.header("Twitter Analysis")
    st.write("""
    This app specializes in analyzing tweet sentiment using an LSTM model.
    It handles Twitter-specific elements like hashtags and mentions.
    """)
    st.markdown("---")
    st.write("**Twitter Features Supported:**")
    st.write("- Hashtags (#)")
    st.write("- Mentions (@)")
    st.write("- Short forms & slang")
    
    st.markdown("---")
    st.write("**Model Info:**")
    st.write("- LSTM Neural Network")
    st.write("- Trained on Twitter data")
    st.write("- Optimized for social media text")

# Load model and tokenizer
@st.cache_resource
def load_components():
    try:
        # Load the LSTM model
        model = load_model('lstm_sentiment_model.keras')
        
        # Load the tokenizer
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

model, tokenizer = load_components()

# Twitter text preprocessing
def preprocess_tweet(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\'.,!?;:]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Function to analyze sentiment
def analyze_tweet_sentiment(text, model, tokenizer, max_len=50):
    if model is None or tokenizer is None:
        return "Error", 0.0, None, text
    
    # Preprocess the tweet
    processed_text = preprocess_tweet(text)
    
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
    
    return sentiment, confidence, probabilities, processed_text

# Main app
text_input = st.text_area("Enter a tweet or Twitter text:", height=100, 
                         placeholder="Paste a tweet here...", key="tweet_input")

if st.button("Analyze Tweet 🚀"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    elif model is None:
        st.error("Model could not be loaded. Please check your model files.")
    else:
        with st.spinner("Analyzing tweet with LSTM... 🤖"):
            # Progress bar
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Get prediction
            sentiment, confidence, probabilities, processed_text = analyze_tweet_sentiment(text_input, model, tokenizer)
            
            # Display results
            st.success("Tweet analysis complete! 🎉")
            
            # Columns for layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Sentiment icon and confidence
                st.markdown("### Sentiment")
                if sentiment == "Positive":
                    st.markdown("<h1 style='text-align: center; color: #17BF63;'>😊</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div class='positive'><h3 style='color: #17BF63;'>Positive</h3><p>Confidence: {confidence:.0%}</p></div>", unsafe_allow_html=True)
                elif sentiment == "Negative":
                    st.markdown("<h1 style='text-align: center; color: #E0245E;'>😠</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div class='negative'><h3 style='color: #E0245E;'>Negative</h3><p>Confidence: {confidence:.0%}</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<h1 style='text-align: center; color: #657786;'>😐</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div class='neutral'><h3 style='color: #657786;'>Neutral</h3><p>Confidence: {confidence:.0%}</p></div>", unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown("### Confidence Level")
                meter_color = "#17BF63" if sentiment == "Positive" else "#E0245E" if sentiment == "Negative" else "#657786"
                meter_html = f"""
                <div style="background: #E1E8ED; width: 100%; height: 20px; border-radius: 10px;">
                    <div style="background: {meter_color}; width: {confidence*100}%; height: 20px; border-radius: 10px;">
                    </div>
                </div>
                <p style="text-align: right; margin-top: 5px;">{confidence:.0%}</p>
                """
                st.markdown(meter_html, unsafe_allow_html=True)
            
            with col2:
                # Sentiment probabilities chart
                st.markdown("### Sentiment Probabilities")
                data = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Probability': probabilities
                })
                
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = ['#E0245E', '#657786', '#17BF63']
                ax.bar(data['Sentiment'], data['Probability'], color=colors)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Tweet Sentiment Probabilities', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Display processed text
            with st.expander("View processed tweet text"):
                st.write("**Original tweet:**")
                st.write(text_input)
                st.write("**After preprocessing:**")
                st.code(processed_text)
            
            # Twitter insights
            st.markdown("### Twitter Insights 🧠")
            if sentiment == "Positive":
                st.info("""
                **Positive tweets** like this often:
                - Contain uplifting or happy content
                - May have exclamation points for emphasis
                - Often include positive words
                """)
            elif sentiment == "Negative":
                st.info("""
                **Negative tweets** like this often:
                - Express complaints or frustrations
                - Sometimes use ALL CAPS for emphasis
                - Often include negative words
                """)
            else:
                st.info("""
                **Neutral tweets** like this often:
                - Share factual information or news
                - May be objective statements
                - Frequently lack strong emotional language
                """)

# Footer with Twitter theme
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #1DA1F2;">
    <p style="font-weight: bold;">Twitter Sentiment Analysis App</p>
    <p>Analyze the emotions behind tweets with AI</p>
    <p>Author: C.Gyireh</p>
</div>
""", unsafe_allow_html=True)
