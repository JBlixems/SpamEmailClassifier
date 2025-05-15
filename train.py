import csv
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

def restore_commas(text):
    """
    Replace the placeholder '|' back to commas in the text.
    """
    return text.replace('|', ',')

def load_data(train_file, test_file):
    """
    Load train and test data from processed CSV files, restoring commas in the text.
    """
    print(f"\nLoading training data from: {train_file}")
    train_df = pd.read_csv(train_file, quotechar='"', engine='python')
    train_df['text_combined'] = train_df['text_combined'].apply(restore_commas)
    print(f"  Loaded {len(train_df)} training rows")

    print(f"\nLoading testing data from: {test_file}")
    test_df = pd.read_csv(test_file, quotechar='"', engine='python')
    test_df['text_combined'] = test_df['text_combined'].apply(restore_commas)
    print(f"  Loaded {len(test_df)} testing rows")

    return train_df, test_df

def preprocess(df):
    """
    Preprocess the data: use the 'text_combined' field directly.
    """
    print("\nPreprocessing data: using 'text_combined' field as text feature")
    X = df['text_combined']
    y = df['label']
    print(f"  Number of samples: {len(X)}")
    return X, y

def train_and_evaluate(train_file, test_file, model_path):
    print("\n=== Training and Evaluation Pipeline ===")
    
    train_df, test_df = load_data(train_file, test_file)

    X_train, y_train = preprocess(train_df)
    X_test, y_test = preprocess(test_df)

    print("\nVectorizing text data with TF-IDF")
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    print(f"  Training vector shape: {X_train_vec.shape}")
    X_test_vec = vectorizer.transform(X_test)
    print(f"  Test vector shape: {X_test_vec.shape}")

    print("\nTraining Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=100000)
    clf.fit(X_train_vec, y_train)
    print("  Training complete.")

    print("\nEvaluating model on test set...")
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nSaving model and vectorizer to {model_path}...")
    joblib.dump({'vectorizer': vectorizer, 'model': clf}, model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == '__main__':
    train_and_evaluate("ProcessedSet/train.csv", "ProcessedSet/test.csv", "spam_model.joblib")