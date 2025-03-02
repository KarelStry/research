'''
    CWRU Quants Research Vertical Exposure Project
'''

# Import statements
import xml.etree.ElementTree as ET
import re
import os

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


def extract_exposure2(seed_words, text_string, buffer):
    """
    Extracts regions around seed words and their similar words using KeyBERT.
    
    Args:
        seed_words (list): List of seed words to search for in the text.
        text_string (str): The text to analyze.
        buffer (int): gives the number of words to extract left and right of the considered word
        
    Returns:
        dict: Dictionary with seed words and similar words as keys, 
              and the surrounding words as values.
    """


    all_words = re.findall(r'\b\w+\b', text_string.lower())

    kw_model = KeyBERT()

    # Use KeyBERT to extract related words based on the full text
    keywords = kw_model.extract_keywords(text_string, keyphrase_ngram_range=(1, 2), 
                                         stop_words='english', top_n=10)
    
    # Extract just the words from the KeyBERT results
    similar_words = set(word.lower() for word, _ in keywords)
    
    # Include both seed words and their similar words
    search_words = set(seed_words) | similar_words

    results = {}

    for i, word in enumerate(all_words):
        normalized_word = word.strip()
        if normalized_word in search_words:
            start_idx, end_idx = max(0, i - buffer), min(len(all_words), i + buffer + 1)
            surrounding_words = ' '.join(all_words[start_idx:end_idx])

            if normalized_word not in results:
                results[normalized_word] = []

            results[normalized_word].append(surrounding_words)

    return results

def sentiment_score():
    """
        Returns sentiment score using roBERTa method for positive/negative/neutral sentiment surrounding our exposure words.
    """

    return 0


