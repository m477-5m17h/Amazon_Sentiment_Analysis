import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    """
    Preprocesses the input text by removing punctuation, converting to lowercase,
    tokenizing, and removing stopwords.

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    # Check if the text is NaN (a float value in pandas)
    if pd.isna(text):
        return ""

    # Ensure the text is treated as a string
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Tokenization and removing stop words
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)
