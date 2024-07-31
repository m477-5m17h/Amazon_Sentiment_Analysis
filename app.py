from flask import Flask, render_template, send_from_directory

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    """
    This is the home route which renders the index page.
    It provides an overview and links to various visualizations.
    """
    return render_template('index.html')

@app.route('/visualizations/<filename>')
def visualizations(filename):
    """
    This route serves the visualization images from the static/images directory.
    """
    return send_from_directory('static/images', filename)

# Routes for each type of visualization
@app.route('/sentiment_distribution')
def sentiment_distribution():
    """
    This route renders the sentiment distribution visualization.
    """
    return render_template('visualization.html', image='sentiment_distribution.png', title='Sentiment Distribution')

@app.route('/positive_wordcloud')
def positive_wordcloud():
    """
    This route renders the word cloud for positive sentiments.
    """
    return render_template('visualization.html', image='positive_wordcloud.png', title='Positive Word Cloud')

@app.route('/negative_wordcloud')
def negative_wordcloud():
    """
    This route renders the word cloud for negative sentiments.
    """
    return render_template('visualization.html', image='negative_wordcloud.png', title='Negative Word Cloud')

@app.route('/neutral_wordcloud')
def neutral_wordcloud():
    """
    This route renders the word cloud for neutral sentiments.
    """
    return render_template('visualization.html', image='neutral_wordcloud.png', title='Neutral Word Cloud')

@app.route('/positive_adjectives')
def positive_adjectives():
    """
    This route renders the common adjectives plot for positive sentiments.
    """
    return render_template('visualization.html', image='positive_adjectives.png', title='Positive Adjectives')

@app.route('/negative_adjectives')
def negative_adjectives():
    """
    This route renders the common adjectives plot for negative sentiments.
    """
    return render_template('visualization.html', image='negative_adjectives.png', title='Negative Adjectives')

@app.route('/neutral_adjectives')
def neutral_adjectives():
    """
    This route renders the common adjectives plot for neutral sentiments.
    """
    return render_template('visualization.html', image='neutral_adjectives.png', title='Neutral Adjectives')

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
