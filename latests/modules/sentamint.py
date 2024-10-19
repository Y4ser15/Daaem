import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import openai
from tqdm import tqdm
import time

# Set your OpenAI API key
# openai.api_key = "#"


def analyze_sentiment_openai(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert. Analyze the sentiment of the following Arabic text and respond with a single number: 1 for positive, 0 for neutral, -1 for negative.",
                },
                {"role": "user", "content": text},
            ],
        )
        return int(response.choices[0].message["content"].strip())
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return None


def process_data(data):
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # Analyze sentiment for each text
    sentiments = []
    for text in tqdm(df["text"], desc="Analyzing sentiments"):
        sentiment = analyze_sentiment_openai(text)
        sentiments.append(sentiment)
        time.sleep(1)  # To avoid hitting rate limits

    df["sentiment"] = sentiments
    return df


def create_sentiment_over_time_plot(df):
    plt.figure(figsize=(12, 6))
    df.set_index("date").resample("D")["sentiment"].mean().plot()
    plt.title("Average Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.show()


def create_sentiment_distribution_plot(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x="sentiment", data=df, order=[-1, 0, 1])
    plt.title("Distribution of Sentiment Scores")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Count")
    plt.xticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
    plt.show()


def create_word_cloud(df):
    text = " ".join(df["text"])
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        font_path="/path/to/arabic/font.ttf",
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Reviews")
    plt.show()


def analyze_top_keywords(df, n=10):
    words = " ".join(df["text"]).split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[word[1] for word in top_words], y=[word[0] for word in top_words])
    plt.title(f"Top {n} Keywords")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()


# Main execution
if __name__ == "__main__":
    # Sample data (replace with your actual scraped data)
    scraped_data = [
        {"text": "الطعام رائع والخدمة ممتازة!", "date": "2024-01-01"},
        {"text": "تجربة مخيبة للآمال، الخدمة بطيئة.", "date": "2024-01-02"},
        # ... more data ...
    ]

    # Process the data
    df = process_data(scraped_data)

    # Create visualizations
    create_sentiment_over_time_plot(df)
    create_sentiment_distribution_plot(df)
    create_word_cloud(df)
    analyze_top_keywords(df)

    # Print summary statistics
    print(df["sentiment"].describe())

    # Analyze sentiment trends
    monthly_sentiment = df.set_index("date").resample("M")["sentiment"].mean()
    print("\nMonthly Sentiment Trends:")
    print(monthly_sentiment)
