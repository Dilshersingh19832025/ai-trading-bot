import requests
from datetime import datetime, timedelta
import csv
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NewsAPI.org API key (replace with your actual API key)
API_KEY = '15124a442b3b4da4bf5a8324d7633f81'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

def fetch_news(query, from_date, to_date, language='en', sort_by='publishedAt', page_size=100):
    """Fetch news articles from NewsAPI."""
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': language,
        'sortBy': sort_by,
        'pageSize': page_size,
        'apiKey': API_KEY
    }
    print(f"Fetching news with query: {query}, from: {from_date}, to: {to_date}")  # Debugging
    try:
        response = requests.get(NEWS_API_URL, params=params)
        print(f"Response Status Code: {response.status_code}")  # Debugging
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            print(f"Fetched {len(articles)} articles.")  # Debugging
            return articles
        else:
            print(f"Error fetching news: {response.status_code} - {response.text}")  # Debugging
            return []
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")  # Debugging
        return []

def analyze_sentiment(text):
    """Analyze the sentiment of a given text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def main():
    print("Starting news sentiment analysis...")  # Debugging

    # Define the search query and date range
    query = 'technology'
    to_date = datetime.now()
    from_date = to_date - timedelta(days=7)  # Last 7 days

    # Fetch news articles
    articles = fetch_news(query, from_date.isoformat(), to_date.isoformat())

    if not articles:
        print("No articles found.")  # Debugging
        return

    print(f"Analyzing sentiment for {len(articles)} articles...")  # Debugging

    # Save results to a CSV file
    with open('news_sentiment_results.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Sentiment Compound", "Sentiment Negative", "Sentiment Neutral", "Sentiment Positive"])

        # Analyze sentiment for each article
        compound_scores = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')

            # Combine title, description, and content for sentiment analysis
            text = f"{title}. {description}. {content}"
            sentiment = analyze_sentiment(text)
            compound_scores.append(sentiment['compound'])

            # Write results to CSV
            writer.writerow([title, sentiment['compound'], sentiment['neg'], sentiment['neu'], sentiment['pos']])

            # Print only strongly positive articles (compound score > 0.5)
            if sentiment['compound'] > 0.5:
                print(f"Title: {title}")
                print(f"Sentiment Compound: {sentiment['compound']}")
                print(f"Sentiment Negative: {sentiment['neg']}")
                print(f"Sentiment Neutral: {sentiment['neu']}")
                print(f"Sentiment Positive: {sentiment['pos']}")
                print("-" * 60)

    print("Sentiment analysis results saved to 'news_sentiment_results.csv'.")  # Debugging

    # Plot sentiment distribution
    plt.hist(compound_scores, bins=20, color='blue', alpha=0.7)
    plt.title("Sentiment Compound Score Distribution")
    plt.xlabel("Compound Score")
    plt.ylabel("Frequency")
    plt.savefig('sentiment_distribution.png')  # Save the plot as an image
    print("Sentiment distribution plot saved to 'sentiment_distribution.png'.")  # Debugging

if __name__ == "__main__":
    main()