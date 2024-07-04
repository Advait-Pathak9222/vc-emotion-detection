import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# logging configuration
logger = logging.getLogger('data_transformation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def download_nltk_resources():
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        logger.debug('NLTK resources downloaded successfully')
    except Exception as e:
        logger.error('Error downloading NLTK resources: %s', e)
        raise

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        logger.debug('Converted to lower case')
        df['content'] = df['content'].apply(remove_stop_words)
        logger.debug('Stop words removed')
        df['content'] = df['content'].apply(removing_numbers)
        logger.debug('Numbers removed')
        df['content'] = df['content'].apply(removing_punctuations)
        logger.debug('Punctuations removed')
        df['content'] = df['content'].apply(removing_urls)
        logger.debug('URLs removed')
        df['content'] = df['content'].apply(lemmatization)
        logger.debug('Lemmatization performed')
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main():
    try:
        download_nltk_resources()

        # Fetch the data from data/raw
        logger.debug('Loading train data from ./data/raw/train.csv')
        train_data = pd.read_csv('./data/raw/train.csv')
        logger.debug('Loading test data from ./data/raw/test.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        logger.debug('Starting normalization of train data')
        train_processed_data = normalize_text(train_data)
        logger.debug('Normalization of train data completed')

        logger.debug('Starting normalization of test data')
        test_processed_data = normalize_text(test_data)
        logger.debug('Normalization of test data completed')

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        logger.debug('Saving processed train data to %s', os.path.join(data_path, "train_processed.csv"))
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        logger.debug('Saving processed test data to %s', os.path.join(data_path, "test_processed.csv"))
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
