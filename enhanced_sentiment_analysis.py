import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sentiment analysis results from the CSV file
try:
    df = pd.read_csv('news_sentiment_results.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'news_sentiment_results.csv' not found. Please ensure the file exists in the correct directory.")
    exit()

# 1. Filter articles based on sentiment thresholds
def filter_articles(df, threshold_positive=0.5, threshold_negative=-0.5):
    highly_positive = df[df['Sentiment Compound'] > threshold_positive]
    highly_negative = df[df['Sentiment Compound'] < threshold_negative]

    print(f"Highly Positive Articles (Sentiment > {threshold_positive}):")
    print(highly_positive[['Title', 'Sentiment Compound']].to_string(index=False))
    print(f"\nHighly Negative Articles (Sentiment < {threshold_negative}):")
    print(highly_negative[['Title', 'Sentiment Compound']].to_string(index=False))

    return highly_positive, highly_negative

# 2. Compare sentiment trends over time (if multiple runs are available)
def compare_sentiment_trends(df_list):
    trend_data = []
    for i, df in enumerate(df_list):
        avg_sentiment = df['Sentiment Compound'].mean()
        trend_data.append({'Run': i + 1, 'Average Sentiment': avg_sentiment})

    trend_df = pd.DataFrame(trend_data)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=trend_df, x='Run', y='Average Sentiment', marker='o')
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Run Number')
    plt.ylabel('Average Sentiment Score')
    plt.savefig('sentiment_trends_over_time.png')
    plt.show()

# 3. Refine the query to focus on specific topics
def refine_query(df, topic=None):
    if topic:
        df = df[df['Title'].str.contains(topic, case=False)]
    return df

# Main execution
if __name__ == "__main__":
    # Filter articles based on sentiment thresholds
    print("Filtering articles...")
    highly_positive, highly_negative = filter_articles(df)
    print("Articles filtered.")

    # Save filtered results to CSV files
    highly_positive.to_csv('highly_positive_articles.csv', index=False)
    highly_negative.to_csv('highly_negative_articles.csv', index=False)
    print("Highly positive articles saved to 'highly_positive_articles.csv'.")
    print("Highly negative articles saved to 'highly_negative_articles.csv'.")

    # Compare sentiment trends (if multiple runs are available)
    print("Comparing sentiment trends...")
    try:
        df_previous_run = pd.read_csv('previous_news_sentiment_results.csv')
        compare_sentiment_trends([df, df_previous_run])
    except FileNotFoundError:
        print("No previous sentiment analysis results found for comparison.")

    # Refine the query to focus on specific topics
    print("Refining query...")
    refined_df = refine_query(df, topic="AI")
    print("Query refined.")

    # Save refined results to a CSV file
    refined_df.to_csv('refined_articles.csv', index=False)
    print("Refined articles saved to 'refined_articles.csv'.")
