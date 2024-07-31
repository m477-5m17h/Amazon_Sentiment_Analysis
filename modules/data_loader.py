import pandas as pd


def load_data(amazon_reviews):
    """
    Loads the dataset from the specified file path.

    Parameters:
    amazon_reviews (str): The file path to the Amazon reviews dataset.

    Returns:
    DataFrame: A pandas DataFrame containing the Amazon reviews data.
    """
    return pd.read_csv(amazon_reviews)
