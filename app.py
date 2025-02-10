import tweepy
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

# Twitter API credentials
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

# Authenticate with Twitter API
try:
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    st.success("Twitter API connection successful!")
except Exception as e:
    st.error(f"Failed to connect to Twitter API: {e}")

# Watch brands to analyze
brands = ["Omega", "Breitling", "Tag Heuer", "Cartier", "Rolex", "Longines", "Rado", 
          "Tissot", "Hublot", "Patek Philippe", "Swatch", "Chopard", "Ulysse Nardin"]

# Scrape Twitter Data
def scrape_twitter_data(brand, count=100):
    try:
        tweets = api.search_tweets(q=brand, count=count, lang="en", tweet_mode="extended")
        data = [[brand, tweet.full_text, tweet.favorite_count] for tweet in tweets]
        return data
    except Exception as e:
        st.error(f"Error fetching Twitter data for {brand}: {e}")
        return []

# Collect data for all brands with live feedback
all_data = []
for brand in brands:
    st.write(f"Fetching tweets for: {brand}...")
    all_data.extend(scrape_twitter_data(brand))

# Convert to DataFrame
if all_data:
    df = pd.DataFrame(all_data, columns=['Brand', 'Text', 'Likes'])

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r'[^A-Za-z ]+', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(tokens)

    # Apply text cleaning
    df['Cleaned_Text'] = df['Text'].apply(clean_text)

    # Sentiment Analysis
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    df['Sentiment_Score'] = df['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['Sentiment'] = df['Sentiment_Score'].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

    # Streamlit Web App
    st.title("Watch Brand Sentiment Analysis - Twitter")

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
    df.to_csv("watch_brand_sentiment_twitter.csv", index=False)
else:
    st.error("No data fetched. Please check the Twitter API or brand names.")
