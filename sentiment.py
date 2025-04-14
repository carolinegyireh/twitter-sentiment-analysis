import streamlit as st
import pandas as pd
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
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
st.title("😊😠😐 Sentiment Analysis Tool")
st.markdown("""
<div style="background-color:#FFDAC1;padding:10px;border-radius:10px;margin-bottom:20px;">
    <h3 style="color:#E27D60;text-align:center;">Discover the emotions behind your text!</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header("About This App")
    st.write("""
    This app analyzes the sentiment of your text using a trained machine learning model. 
    It can detect positive 😊, negative 😠, or neutral 😐 sentiments.
    """)
    st.markdown("---")
    st.write("Made with ❤️ for sentiment analysis")
    
    # Add a cute image (you can replace with your own)
    image = Image.open("sentiment_image.png")  # Create a simple image or remove this line
    st.image(image, caption="Understanding Emotions", use_column_width=True)

# Function to analyze sentiment (replace with your actual model)
def analyze_sentiment(text):
    # This is a placeholder - replace with your model's prediction code
    time.sleep(1)  # Simulate processing time
    
    # For demo purposes - random sentiment
    # Replace this with your model's prediction
    sentiment = np.random.choice(["Positive", "Negative", "Neutral"], p=[0.4, 0.3, 0.3])
    confidence = np.random.uniform(0.7, 0.98)
    
    return sentiment, confidence

# Main app
text_input = st.text_area("Enter your text here:", height=150, 
                         placeholder="Type something like 'I love this product! It's amazing!'")

if st.button("Analyze Sentiment 🚀"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing your text... 🤖"):
            # Progress bar for cuteness
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Get prediction (replace with your model)
            sentiment, confidence = analyze_sentiment(text_input)
            
            # Display results
            st.success("Analysis complete! 🎉")
            
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
            
            # Sentiment distribution visualization (demo)
            st.markdown("### Sentiment Distribution")
            data = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Value': [0.4, 0.3, 0.3]  # Replace with actual probabilities from your model
            })
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Sentiment', y='Value', data=data, palette=['#FF9AA2', '#FFB7B2', '#FFDAC1'], ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            plt.title('Predicted Sentiment Probabilities')
            st.pyplot(fig)
            
            # Fun facts based on sentiment
            st.markdown("### Fun Fact 💡")
            if sentiment == "Positive":
                st.info("Did you know? Positive people live longer and have stronger immune systems!")
            elif sentiment == "Negative":
                st.info("Expressing negative emotions can be healthy - it's better out than in!")
            else:
                st.info("Neutral is the new black - sometimes balance is the best approach!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #E27D60;'>Thank you for using our Sentiment Analysis App! ❤️</p>", 
            unsafe_allow_html=True)
