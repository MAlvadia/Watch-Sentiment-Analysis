import praw
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Reddit API with error handling
try:
    reddit = praw.Reddit(
        client_id=os.getenv("CLIENT_ID"),       # Reddit client ID
        client_secret=os.getenv("CLIENT_SECRET"), # Reddit client secret
        user_agent=os.getenv("USER_AGENT")       # Reddit user agent
    )
    reddit.read_only = True  # Verify connection
    st.success("Reddit API connection successful!")
except Exception as e:
    st.error(f"Failed to connect to Reddit API: {e}")

# Watch brands to analyze
brands = ["Omega", "Breitling", "Tag Heuer", "Cartier", "Rolex", "Longines", "Rado", 
          "Tissot", "Hublot", "Patek Philippe", "Swatch", "Chopard", "Ulysse Nardin"]

# Scrape Reddit Data
def scrape_reddit_data(brand, subreddit="Watches", limit=50):
    try:
        subreddit = reddit.subreddit(subreddit)
        posts = subreddit.search(brand, limit=limit)
        
        data = []
        for post in posts:
            data.append([brand, post.title, post.selftext, post.score])
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {brand}: {e}")
        return []

# Collect data for all brands with live feedback
all_data = []
for brand in brands:
    st.write(f"Fetching data for: {brand}...")
    all_data.extend(scrape_reddit_data(brand))

# Convert to DataFrame
if all_data:
    df = pd.DataFrame(all_data, columns=['Brand', 'Title', 'Text', 'Upvotes'])

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r'[^A-Za-z ]+', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(tokens)

    # Apply text cleaning
    df['Cleaned_Text'] = df['Title'] + " " + df['Text']
    df['Cleaned_Text'] = df['Cleaned_Text'].apply(clean_text)

    # Sentiment Analysis
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    df['Sentiment_Score'] = df['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['Sentiment'] = df['Sentiment_Score'].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

    # Streamlit Web App
    st.title("Watch Brand Sentiment Analysis")

    selected_brand = st.selectbox("Select a Brand", brands)

    st.write("### Sentiment Distribution")
    sentiment_counts = df[df['Brand'] == selected_brand]['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.write("### Word Cloud")
    text = " ".join(df[df['Brand'] == selected_brand]['Cleaned_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.image(wordcloud.to_array(), use_column_width=True)

    st.write("### Raw Data")
    st.dataframe(df[df['Brand'] == selected_brand])

    # Save Data
    df.to_csv("watch_brand_sentiment.csv", index=False)
else:
    st.error("No data fetched. Please check the Reddit API or brand names.")

