from modules.data_loader import load_data
from modules.text_preprocessing import preprocess_text
from modules.visualizations import plot_sentiment_distribution, generate_wordclouds, plot_common_adjectives
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from imblearn.over_sampling import SMOTE
import joblib


def preprocess_and_save():
    """
    This function is the heart of our application. It preprocesses the Amazon review data,
    trains a Support Vector Machine (SVM) model to classify sentiments, and generates visualizations
    to help understand customer feedback.

    References:
    - Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
    """
    dataset_path = 'data/amazon_reviews.csv'

    # Load the data
    df = load_data(dataset_path)

    # Preprocess the review text
    df['processed_reviewText'] = df['reviewText'].apply(preprocess_text)

    # Label sentiments based on review ratings
    def label_sentiment(row):
        if row['overall'] > 3:
            return 'positive'
        elif row['overall'] < 3:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment'] = df.apply(lambda row: label_sentiment(row), axis=1)

    # Convert text data to numerical data using TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(df['processed_reviewText']).toarray()
    y = pd.Categorical(df['sentiment']).codes

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train the SVM model
    svm_model = SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=42)
    svm_model.fit(X_train_res, y_train_res)

    # Save the model and vectorizer for future use
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    # Generate and save visualizations
    plot_sentiment_distribution(df)
    generate_wordclouds(df)
    plot_common_adjectives(df)

    # Evaluate the model and print the results
    predictions = svm_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n",
          classification_report(y_test, predictions, target_names=['negative', 'neutral', 'positive']))


if __name__ == "__main__":
    preprocess_and_save()
