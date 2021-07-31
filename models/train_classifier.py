import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    """
    Loads data from the Disaster table into a dataframe also split the data between 
    labels and features
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disaster' , engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns.values
    return X, y, category_names


def tokenize(text):
    """
    Break the text into clean tokens, using lemmatization
    and normalize the data.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Build a model pipeline and select the best params to the model
    using grid search.
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'clf__estimator__min_samples_split': [ 3, 4]
    }

    model = GridSearchCV(pipeline, param_grid= parameters, cv=2, n_jobs= -1, verbose= 10)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model created among accuracy, test, recall and f1 metrics.
    """
    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred, target_names= category_names)
    print(class_report)


def save_model(model, model_filepath):
    #Save the model to a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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