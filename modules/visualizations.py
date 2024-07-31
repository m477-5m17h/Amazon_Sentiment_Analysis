import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.probability import FreqDist

# Download necessary NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

def plot_sentiment_distribution(df):
    """
    Generates and saves a bar plot showing the distribution of sentiments in the dataset.
    """
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 8))
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    file_path = 'static/images/sentiment_distribution.png'
    plt.savefig(file_path)  # Save the figure
    plt.close()
    print(f"Saved sentiment distribution plot to {file_path}")

def generate_wordclouds(df):
    """
    Generates and saves word clouds for each sentiment category.
    """
    sentiments = ['positive', 'negative', 'neutral']
    for sentiment in sentiments:
        text = " ".join(review for review in df[df["sentiment"] == sentiment]['processed_reviewText'])
        wordcloud = WordCloud(background_color="white").generate(text)
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {sentiment.capitalize()} Sentiment')
        plt.axis("off")
        file_path = f'static/images/{sentiment}_wordcloud.png'
        plt.savefig(file_path)  # Save the figure
        plt.close()
        print(f"Saved word cloud for {sentiment} sentiment to {file_path}")

def get_adjectives(text):
    """
    Tokenizes the text, removes stopwords, and returns adjectives.
    """
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tagged = pos_tag(tokens)
    adjectives = [word for word, tag in tagged if tag in ['JJ', 'JJR', 'JJS']]
    return adjectives

def plot_common_adjectives(df):
    """
    Generates and saves bar plots showing the most common adjectives for each sentiment category.
    """
    sentiments = ['positive', 'negative', 'neutral']
    for sentiment in sentiments:
        text = " ".join(df[df['sentiment'] == sentiment]['processed_reviewText'])
        adjectives = get_adjectives(text)
        adj_freq = FreqDist(adjectives)
        adj_df = pd.DataFrame(adj_freq.most_common(10), columns=['Adjective', 'Frequency'])

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Frequency', y='Adjective', data=adj_df, color='blue')
        plt.title(f'Most Common Adjectives in {sentiment.capitalize()} Reviews')
        plt.xlabel('Frequency')
        plt.ylabel('Adjective')
        file_path = f'static/images/{sentiment}_adjectives.png'
        plt.savefig(file_path)  # Save the figure
        plt.close()
        print(f"Saved common adjectives plot for {sentiment} sentiment to {file_path}")
