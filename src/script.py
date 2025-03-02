from functions import extract_text, extract_exposure, csv_to_list, sentiment_score
import json
import os
from pathlib import Path
import logging

def model5(analyze_path, exposure_csv, n):
    """
        Model 5 Pipeline:
        - text extraction from earnings call
        - exposure csv to exposure word list
        - exposure search with +- parameter
        - sentiment analysis on found exposure

        Returns:
            dict with exposure strings, sentiment score, pos/neg
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Extracting Text...")
    text = extract_text(analyze_path)

    logging.info("Loading Exposure Word List...")
    exposure_word_list = csv_to_list(exposure_csv)

    logging.info("Calculating Exposure...")
    exposure = extract_exposure(text, exposure_word_list, window=n)

    logging.info("Finding Sentiment...")
    final = sentiment_score(exposure)

    return final

def model5_f(folder_path, exposure_csv, buffer, output_file):
    """
    Processes a folder of earnings call transcripts, applies Model 5, and saves results in a JSON file.

    Inputs:
    - folder_path: str, path to the folder containing XML earnings call transcripts.
    - exposure_csv: str, comma-separated exposure words.
    - buffer: int, number of words before and after the exposure word.
    - output_file: str, JSON file to save the results.

    Output:
    - A JSON file containing:
        - File name
        - Total count of exposure instances
        - Positive count & percentage
        - Neutral count & percentage
        - Negative count & percentage
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    results = {}

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xml"):
            file_path = os.path.join(folder_path, file_name)
            logging.info(f"Processing {file_name}...")

            try:
                # run Model 5 pipeline
                exposure_dict = model5(file_path, exposure_csv, buffer)

                # occurence count
                total_count = len(exposure_dict)
                positive_count = sum(1 for v in exposure_dict.values() if v["label"] == "positive")
                neutral_count = sum(1 for v in exposure_dict.values() if v["label"] == "neutral")
                negative_count = sum(1 for v in exposure_dict.values() if v["label"] == "negative")

                # percentage calc
                positive_percentage = round((positive_count / total_count) * 100, 2) if total_count > 0 else 0
                neutral_percentage = round((neutral_count / total_count) * 100, 2) if total_count > 0 else 0
                negative_percentage = round((negative_count / total_count) * 100, 2) if total_count > 0 else 0

                # save
                results[file_name] = {
                    "total_count": total_count,
                    "positive_count": positive_count,
                    "positive_percentage": positive_percentage,
                    "neutral_count": neutral_count,
                    "neutral_percentage": neutral_percentage,
                    "negative_count": negative_count,
                    "negative_percentage": negative_percentage,
                }

                logging.info(f"Finished processing {file_name}")

            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")
        count += 1
        if count > 10:
            break

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Results saved to {output_file}")



if __name__ == "__main__":
    from functions import csv_to_list
    # j = model5("/Users/efang/Desktop/coding/research/src/data/example.xml", "/Users/efang/Desktop/coding/research/src/data/political_words_extended.csv", 20)
    exposure_folder = "/Users/efang/Desktop/coding/research/src/data/political_words_extended.csv"
    folder_path = "/Users/efang/Downloads/Transcript/2016"
    
    model5_f(folder_path, exposure_folder, 20, "trial1.json")
    

