'''
    CWRU Quants Research Vertical Exposure Project
'''

# Import statements
import xml.etree.ElementTree as ET
import re
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer 
from bs4 import BeautifulSoup

# Functions
def extract_text(file_path: str) -> str:
    """

    This function takes in an earnings transcript as an input, and extracts the words from the transcript.

    Input: .xml file
    Output: str
    """
    
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract all text content from XML elements, including CDATA
    text_parts = []
    
    # Look specifically for Body and Headline elements which contain CDATA
    for event_story in root.findall('.//EventStory'):
        # Extract headline
        headline = event_story.find('Headline')
        if headline is not None and headline.text:
            text_parts.append(headline.text.strip())
        
        # Extract body
        body = event_story.find('Body')
        if body is not None and body.text:
            text_parts.append(body.text.strip())
    
    # Join all text parts with spaces
    raw_text = ' '.join(text_parts)
    
    # Clean up the text
    # Remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', raw_text)
    # Remove special characters and normalize whitespace
    cleaned_text = re.sub(r'[^\w\s.,?!-]', '', cleaned_text)
    # Trim leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def csv_to_list(filepath):
    """
    This takes in a CSV file consisting of political bigrams and returns
    a list of bigrams with the underscore removed (e.g., "democratic_party" -> "democratic party").
    """
    political_list = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Convert the CSV rows to a list so we can manipulate them outside the 'with' block
        rows = list(reader)
    
    # Remove the first row if it's a header
    if rows:
        rows.pop(0)
    
    # Process each row, replacing underscores with spaces
    for row in rows:
        # Ensure row is not empty
        if row and len(row) > 0:
            bigram = row[0]
            # Replace the first underscore with a space if you only expect two parts
            parts = bigram.split("_")
            if len(parts) == 2:
                bigram = parts[0] + " " + parts[1]
            political_list.append(bigram.lower())
    
    return political_list


def extract_exposure(text, keywords, window=10) -> dict :
    """
        This takes in the str returned from extract_text, and extracts regions (+- buffer) where the exposure
        words exist. 

        For example, if buffer = 5, then wherever we identify an exposure word, we take the substring of words beginning
        5 words before exposure word, and 5 words after the exposure words. This would create a string with 11 words. 
        We would then add this to our return dict.

        Input: 
        - exposure_words: csv
        - txt_string: str
        - buffer: int

        Output:
        dictionary
        
    """
    words = re.findall(r'\w+', text)
    contexts = {}

    for index, word in enumerate(words):
        if word.lower() in keywords:
            start = max(0, index - window)
            end = min(len(words), index + window + 1)
            context = " ".join(words[start:end])
            contexts[word] = context

    return contexts


def sentiment_score(text_dict):
    """
    Returns sentiment scores for each string in text_dict using RoBERTa-based
    sentiment analysis for positive/negative/neutral sentiment.

    TODO:
    - reference to how the sentiment reference works
    """
    from transformers import pipeline
    
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    results = {}

    for key, text in text_dict.items():
        prediction = sentiment_analyzer(text)[0]
        
        label = prediction["label"].lower() 
        score = prediction["score"]     

        if label == "positive":
            numeric_score = score
        elif label == "negative":
            numeric_score = -score
        else:
            numeric_score = 0

        results[key] = {
            "text": text,
            "label": label,
            "score": score,
            "numeric_score": numeric_score
        }

    return results

def tf_idf(*args):
    '''
        Input: 
        Output:
    '''
    documents = list(args)
    if not all(isinstance(doc, str) for doc in documents):
        raise ValueError("All inputs must be strings.")

    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(documents)

    print('\nIDF values:')
    for word, idf in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        print(word, ':', idf)

    return result


