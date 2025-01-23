import re
import pandas as pd
import os
import json
import datetime
import textwrap
import string
import difflib
import glob

def save_to_csv(data, file_path, rewrite=False):
    df_out = pd.DataFrame(data)
    
    # Ensure the directory exists
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    if os.path.exists(file_path) and not rewrite:
        df_out.to_csv(file_path, mode='a', header=False, index=False)  # Append without writing headers
    else:
        df_out.to_csv(file_path, index=False)  # Create new file with headers
    
    print(f"Data saved to {file_path}")

def save_to_csv_with_filepath(data, file_path, rewrite=False):
    df_out = pd.DataFrame(data)
    if os.path.exists(file_path) and not rewrite:
        df_out.to_csv(file_path, mode='a', header=False, index=False)  # Append without writing headers
    else:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        df_out.to_csv(file_path, index=False)  # Create new file with headers
    print(f"Data appended to {file_path}")

def count_csv_entries(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return len(df)
    else:
        return 0
    
def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def count_words(text):
    if text is None:
        return 0
    return len(text.split())

def count_num_of_words(text):
    return len(text.split())

def length_diff_exceeds_percentage(text1, text2, percentage):

    # If less than zero, assume disabled
    if percentage < 0:
        return False

    # Split the texts into words and count the number of words
    len1 = count_num_of_words(text1)
    len2 = count_num_of_words(text2)
    
    # Calculate the absolute difference in the number of words
    word_diff = abs(len1 - len2)
    
    # Calculate the percentage difference relative to the smaller text
    smaller_len = min(len1, len2)
    
    # Avoid division by zero in case one of the texts is empty
    if smaller_len == 0:
        return word_diff > 0
    
    percentage_diff = (word_diff / smaller_len)
    
    # Check if the percentage difference exceeds the specified threshold
    return percentage_diff > percentage, len1, len2

def get_prompt_or_output(csv_path, num):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    # Get the specific text based on num
    if num <= len(df) and num > 0:
        story = df.iloc[num - 1]['text']
    else:
        raise Exception(f"Index out of range.")
    
    return story

def get_prompt_and_id_dev(csv_path, num):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    # Get the specific text based on num
    if num <= len(df) and num > 0:
        story = df.iloc[num - 1]['prompt']
        id = df.iloc[num - 1]['id']
    else:
        raise Exception(f"Index out of range.")
    
    return story, id

def get_prompt_from_id(csv_path, id):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    matching_row = df[df['id'] == id]

    if not matching_row.empty:
        prompt = matching_row.iloc[0]['prompt']
        return prompt
    else:
        raise Exception(f"No match found for ID {id}")

def get_watermarked_text(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df['current_text'].iloc[0]

def get_mutated_text(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    success_df = df[(df['mutated_text_len'] >= 0.95 * df['current_text_len']) & (df['quality_preserved'] == True)]

    return success_df['mutated_text'].iloc[-1]

def get_nth_successful_perturbation(csv_file_path, mutation_num):
    df = pd.read_csv(csv_file_path)
        
    unique_texts = []
    seen_texts = set()  # To track what we've already added

    for current_text in df['current_text']:
        if current_text not in seen_texts:
            unique_texts.append(current_text)
            seen_texts.add(current_text)
    
    return unique_texts[mutation_num]

def get_last_step_num(csv_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Access the last 'step_num' value
    last_step_num = df['step_num'].iloc[-1]
    
    return last_step_num

def strip_up_to(response, delimiter):
    # Find the position of the delimiter
    pos = response.rfind(delimiter)
    
    # If the delimiter is found, return the part of the string after it
    if pos != -1:
        # Adjust the position to remove the delimiter itself
        return response[pos + len(delimiter):].strip()
    return response

def replace_multiple_commas(s):
    # Replace multiple commas with a single comma
    return re.sub(r',+', ',', s)
    
def diff(text1, text2):
    """
    Returns the difference of 2 texts.
    """
    # Splitting the texts into lines as difflib works with lists of lines
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    
    # Creating a Differ object
    d = difflib.Differ()

    # Calculating the difference
    diff = list(d.compare(text1_lines, text2_lines))

    # Joining the result into a single string for display
    diff_result = '\n'.join(diff)

    return diff_result

def read_text_file(file_path):
    """
    Reads a text file and returns its contents as a string.

    Args:
        file_path (str): The path to the text file to be read.

    Returns:
        str: The contents of the file.

    Raises:
        FileNotFoundError: If the file cannot be found at the specified path.
        IOError: If an error occurs during file reading.
    """
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            return contents
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' does not exist.")
        raise
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        raise

def add_prefix_to_keys(original_dict, prefix):
    # Create a new dictionary with the prefix added to each key
    new_dict = {f"{prefix}{key}": value for key, value in original_dict.items()}
    return new_dict

def extract_response_info(sentence):
    # Enhanced regular expression with corrected spacing and flexible matching
    pattern = re.compile(
        r"(response [ab]).*(much better|a little better|better|similar|a little worse|worse|much worse).*?(response [ab])",
        re.IGNORECASE
    )

    # Search for patterns in the sentence
    match = pattern.search(sentence)

    if match:
        response_first = match.group(1).lower()
        comparison = match.group(2).lower()
        if "much" in sentence:
          comparison = "much " + comparison
        elif "a little" in sentence:
          comparison = "a little " + comparison
        response_second = match.group(3).lower()

        # Ensure "response a" is always discussed first in the output
        if response_first.endswith("b"):
            # Reverse the comparison if "response b" is mentioned first
            reverse_comparison_map = {
                "much better": "much worse",
                "a little better": "a little worse",
                "better": "worse",
                "similar": "similar",
                "a little worse": "a little better",
                "worse": "better",
                "much worse": "much better"
            }
            adjusted_comparison = reverse_comparison_map[comparison]
            return ["response a", adjusted_comparison]
        else:
            return ["response a", comparison]
    else:
        return ["", ""]
    

def is_bullet_point(word):
    """
    Checks if the given word is a bullet point in the format '1.', '2.', etc.

    Args:
    word (str): The word to check.

    Returns:
    bool: True if the word is a bullet point, False otherwise.
    """
    # Regular expression pattern to match a digit followed by a period
    pattern = r'^\d+\.$'
    
    # Use re.match to check if the word matches the pattern
    return re.match(pattern, word) is not None

def strip_punct(word):
    """
    Strips punctuation from the left and right of the word and returns a tuple.

    Args:
    word (str): The word to process.

    Returns:
    tuple: A tuple containing the left punctuation, the stripped word, and the right punctuation.
    """
    if not word:  # If the word is empty, return an empty tuple
        return ("", "", "")
    
    # Initialize variables
    left_punctuation = ""
    right_punctuation = ""

    # Strip left punctuation
    i = 0
    while i < len(word) and word[i] in string.punctuation:
        left_punctuation += word[i]
        i += 1
    
    # Strip right punctuation
    j = len(word) - 1
    while j >= 0 and word[j] in string.punctuation:
        right_punctuation = word[j] + right_punctuation
        j -= 1
    
    # The stripped word
    stripped_word = word[i:j+1]

    return (left_punctuation, stripped_word, right_punctuation)

def load_all_csvs(base_dir="./attack/traces", watermark_type="Adaptive", mutator_str="WordMutator"):
    """
    Searches anywhere in the filename for the given `mutator_str`
    (e.g. "WordMutator", "SentenceMutator", etc.) and the `watermark_type`
    (e.g. "GPT4o_unwatermarked", "watermarked", etc.) and loads all
    such CSV files into one DataFrame.

    For instance, if mutator_str = "WordMutator" and watermark_type = "GPT4o_unwatermarked",
    this will match:
        InternLMOracle_GPT4o_unwatermarked_WordMutator_n-steps=1000_attack_results_part6.csv
    and any other CSV that includes the substrings "WordMutator" and "GPT4o_unwatermarked".

    Returns a Pandas DataFrame concatenating all matches.
    If no files match, returns an empty DataFrame.
    """

    # Build a glob pattern that matches anything containing the watermark_type and mutator_str,
    # followed by anything, and ending with .csv
    pattern = os.path.join(base_dir, f"*{watermark_type}*{mutator_str}*annotated*.csv")

    # Get all matching CSV files
    csv_files = glob.glob(pattern)

    # Sort them so part1 < part2 < part3, etc. (if you have chunked files)
    csv_files.sort()

    # Read each CSV into a list of DataFrames
    dataframes = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        dataframes.append(df)

    # Concatenate into one DataFrame
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()
    
def load_all_csvs(base_dir, watermark_str, mutator_str, annotated=True):
    """
    Searches anywhere in the filename for the given `mutator_str`
    (e.g. "WordMutator", "SentenceMutator", etc.) and loads all
    such CSV files into one DataFrame.

    For instance, if mutator_str = "WordMutator", this will match:
        InternLMOracle_GPT4o_unwatermarked_WordMutator_n-steps=1000_attack_results_part6.csv
    and any other CSV that includes the substring "WordMutator".

    Returns a Pandas DataFrame concatenating all matches.
    If no files match, returns an empty DataFrame.
    """

    # Build a glob pattern that matches anything containing the mutator_str,
    # followed by anything, and ending with .csv
    file_paths = f"*?*{watermark_str}_{mutator_str}?*" + ("annotated?*" if annotated else "")
    pattern = os.path.join(base_dir, file_paths)

    # Get all matching CSV files
    csv_files = glob.glob(pattern)

    # Sort them so part1 < part2 < part3, etc. (if you have chunked files)
    csv_files.sort()

    # Read each CSV into a list of DataFrames
    dataframes = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        dataframes.append(df)

    # Concatenate into one DataFrame
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()