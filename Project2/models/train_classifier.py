import sys
#import read and save data libraries
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

#import NPL libraries
import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

#import ML libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Loads data from SQL Database previously created
    Args:
    database_filepath: SQL database file path
    Returns:
    X: features dataframe
    Y: target dataframe
    category_names: target labels 
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con = engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text or message.
    Args:
    text: text
    Returns: array of clean tokens
    """
    # url pattern define
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # find and replace urls
    detected_urls = re.findall(url_re, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")    
    
    #tokenize sentences and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    final_tokens=[]
    for tok in tokens:
        final_tokens = lemmatizer.lemmatize(tok).lower().strip()
    return final_tokens


def build_model():
    """Builds classification model """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #gridsearch
    parameters = {
    'clf__estimator__n_estimators' : [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model
    Args:
    model: Trained model
    X_test: Test features
    Y_test: Test labels
    category_names: labels 
    """
    # predict
    Y_preds = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test.values[:, i], Y_preds[:, i]))    


def save_model(model, model_filepath):
    """
        Save model to pickle
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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