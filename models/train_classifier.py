import sys
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Args:
    database_filepath (str): Filepath for the SQLite database containing the cleaned data.
    
    Returns:
    X (Series): Messages data for training.
    Y (DataFrame): Categories data for training.
    category_names (Index): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = 'DisasterResponse'  # replace with your actual table name
    df = pd.read_sql_table(table_name, con=engine)
    target_columns = df.columns[4:]  # assuming first 4 columns are id, message, original, genre
    df = df.dropna(subset=target_columns)
    X = df['message']
    Y = df[target_columns]
    return X, Y, target_columns

def tokenize(text):
    """
    Tokenize and lemmatize text data.
    
    Args:
    text (str): Text data to be tokenized.
    
    Returns:
    tokens (list): List of cleaned and lemmatized tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_model():
    """
    Build machine learning pipeline and perform grid search.
    
    Returns:
    cv (GridSearchCV): Grid search model object.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__max_df': [0.9, 1.0],
        'tfidf__ngram_range': [(1, 1)],
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on the test set and print out the classification report.
    
    Args:
    model: Trained model.
    X_test (Series): Test data messages.
    Y_test (DataFrame): True labels for test data.
    category_names (Index): List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
    model: Trained model.
    model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to execute the machine learning pipeline: load data, train model, evaluate model, and save model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
