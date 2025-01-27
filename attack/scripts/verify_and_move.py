import pandas as pd
import os
import shutil

REQUIRED_COLUMNS = ["group_id", "watermark_score", "words_edited", 
                   "perplexity", "grammar_errors", "internlm_quality"]

SOURCE_DIR = './attack/traces'
DEST_DIR = './attack/traces/annotated'

def check_csv(file_path):
    """Check if a CSV file meets the required criteria."""
    try:
        # Read CSV header
        df_header = pd.read_csv(file_path, nrows=0)
    except pd.errors.EmptyDataError:
        return False, "Empty file"
    
    # Check required columns
    csv_columns = set(df_header.columns)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in csv_columns]
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    # Check watermark_score has non-null values
    try:
        df_watermark = pd.read_csv(file_path, usecols=['watermark_score'])
        if df_watermark['watermark_score'].notna().sum() == 0:
            return False, "All watermark_score values are null"
    except Exception as e:
        return False, f"Error reading watermark_score: {str(e)}"
    
    return True, "All checks passed"

def main():
    # Ensure destination directory exists
    os.makedirs(DEST_DIR, exist_ok=True)

    # Process all CSV files in source directory
    for filename in os.listdir(SOURCE_DIR):
        if not filename.endswith('.csv'):
            continue
            
        source_path = os.path.join(SOURCE_DIR, filename)
        if not os.path.isfile(source_path):
            continue

        valid, message = check_csv(source_path)
        if valid:
            print(f"Moving {filename}: {message}")
            dest_path = os.path.join(DEST_DIR, filename)
            shutil.move(source_path, dest_path)

if __name__ == '__main__':

    # python -m attack.scripts.verify_and_move

    main()