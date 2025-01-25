import os
import pandas as pd
from itertools import combinations
from dotenv import load_dotenv, find_dotenv
from guidance import models

def extract_unique_column_value(df, column_name):
    """
    Checks if all values in the specified column are identical.
    If so, returns that value. Otherwise, raises an Exception.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to check.
    - column_name (str): The column to verify.
    
    Returns:
    - The unique value in the column.
    
    Raises:
    - Exception: If the column contains multiple unique values.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    unique_count = df[column_name].nunique()
    if unique_count == 1:
        return df[column_name].iloc[0]
    else:
        unique_values = df[column_name].unique()
        raise Exception(f"Column '{column_name}' contains {unique_count} unique values: {unique_values}")

def split_dataframe(df, chunk_size):
    return [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

def get_id_tuples(num=10):
    """
    Generates unique pairs of numbers within groups of three from 1 to `num`.
    Each pair is annotated with the group number it belongs to.

    Parameters:
    num (int): The maximum number to generate pairs up to. Default is 30.

    Returns:
    list of tuples: Each tuple contains two numbers forming a pair and the group number.
    """
    # Initialize an empty list to store the pairs
    pairs = []
    
    # Use enumerate to keep track of the group number, starting at 0
    for group_num, start in enumerate(range(0, num, 3), start=1):
        # Define the current group of three numbers
        group = range(start, start + 3)
        
        # Generate all unique pairs within the group
        for pair in combinations(group, 2):
            # Check if the second number in the pair does not exceed 'num'
            if pair[1] <= num:
                # Append the pair along with the group number to the list
                pairs.append((*pair, group_num))
    
    return pairs

def extract_unique_column_value(df, column_name):
    """
    Checks if all values in the specified column are identical.
    If so, returns that value. Otherwise, raises an Exception.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to check.
    - column_name (str): The column to verify.
    
    Returns:
    - The unique value in the column.
    
    Raises:
    - Exception: If the column contains multiple unique values.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    unique_count = df[column_name].nunique()
    if unique_count == 1:
        return df[column_name].iloc[0]
    else:
        unique_values = df[column_name].unique()
        raise Exception(f"Column '{column_name}' contains {unique_count} unique values: {unique_values}")

def parse_filename(filename):
    
    # Example filename: "{o_str}_{w_str}_{m_str}_n-steps={n_steps}_attack_results.csv"
    # Strip the directory and extension, handle `_attack_results_part` and then split by '_'
    base_name = os.path.basename(filename)
    base_name = base_name.replace('_attack_results.csv', '')
    
    # Handle cases with `_attack_results_part` optionally followed by a string
    if '_attack_results_part' in base_name:
        base_name = base_name.split('_attack_results_part')[0]
    
    parts = base_name.split('_')

    o_str = parts[0]

    if 'GPT' in parts[1]:
        w_str = f"{parts[1]}_{parts[2]}"
        m_str = parts[3]
        n_steps = parts[4].split('=')[1]  # Extract the value after 'n-steps='
    else:
        w_str = parts[1]
        m_str = parts[2]
        n_steps = parts[3].split('=')[1]  # Extract the value after 'n-steps='

    return o_str, w_str, m_str, n_steps

# Helper function to separate attacks based on step_num reset
def separate_attacks(df):
    attacks = []
    current_attack = []
    
    for idx, row in df.iterrows():
        # Start a new attack if the step_num resets
        if idx > 0 and row['step_num'] < df.loc[idx - 1, 'step_num']:
            attacks.append(pd.DataFrame(current_attack))
            current_attack = []
        
        current_attack.append(row)
    
    # Append the last attack
    if current_attack:
        attacks.append(pd.DataFrame(current_attack))
    
    return attacks

def process_attack_traces(directory, filter_func=lambda x: True):
    """
    Processes attack traces from files in a specified directory.

    This function reads CSV files from the given directory and filters them using
    the provided lambda function
    
    Then, it groups files based on a common base name, combines their data, and 
    separates the combined data into individual attacks.

    Parameters:
    directory (str): The path to the directory containing the attack trace files.
    filter_func (function): A lambda function to filter the files in the directory.
                            The function should take a filename as input and return
                            True if the file should be processed, and False otherwise.

    Returns:
    all_attacks: A list of dictionaries, each containing metadata and data for an individual attack.
    """
    all_attacks = []
    file_groups = {}
    
    for filename in os.listdir(directory):
        if not filter_func(filename):
            continue
        base_name = '_'.join(filename.split('_part')[0].split('_')[:-1])
        # log.info(f"Base Name: {base_name}")
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(filename)

    for base_name, files in file_groups.items():
        combined_df = pd.DataFrame()

        for filename in sorted(files):  # Ensure files are processed in order of parts
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Extract variables from the first filename
        o_str, w_str, m_str, n_steps = parse_filename(files[0])
        
        # Separate attacks based on the combined DataFrame
        attacks = separate_attacks(combined_df)
        
        # Store each attack with its metadata
        for i, attack in enumerate(attacks):
            all_attacks.append({
                'o_str': o_str,
                'w_str': w_str,
                'm_str': m_str,
                'n_steps': n_steps,
                'attack_num': i + 1,
                'attack_data': attack
            })
    
    return all_attacks

def get_model(model_name):
    distinguisher_persona = \
    """
    You are an expert in analyzing the similarity of responses.
    You specialize in detecting whether one response has been derived from another by carefully analyzing the content and structure of the text.
    Your expertise in linguistic analysis allows you to distinguish which responses are the most closely related.
    Your goal is to provide a clear, concise, and accurate assessment of the provided instructions.
    # """

    if("llama" in model_name):
        match model_name:
            case "llama3.1-8B":
                model = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf"
            case "llama3.1-70B":
                model = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf"
            case "llama3.3-70B":
                model = "/data2/.shared_models/llama.cpp_models/Llama-3.3-70B-Instruct.Q8_0.gguf"
            case _:
                raise ValueError(f"Model {model_name} not found")

        llm = models.LlamaCpp(
            model=model,
            echo=False,
            n_gpu_layers=-1,
            n_ctx=4096
        )
    elif("gpt" in model_name):
        load_dotenv(find_dotenv())
        llm = models.OpenAI(model_name)

    return llm, distinguisher_persona

def prompt_to_entropy(prompt):
    if "story" in prompt:
        if len(prompt) == 23: return 1
        if len(prompt) == 49: return 2
        if len(prompt) == 65: return 3
        if len(prompt) == 98: return 4
        if len(prompt) == 123: return 5
        if len(prompt) == 209: return 6
        if len(prompt) == 235: return 7
        if len(prompt) == 263: return 8
        if len(prompt) == 340: return 9
        if len(prompt) == 492: return 10
    elif "essay" in prompt:
        if len(prompt) == 65: return 1
        if len(prompt) == 107: return 2
        if len(prompt) == 152: return 3
        if len(prompt) == 194: return 4
        if len(prompt) == 253: return 5
        if len(prompt) == 285: return 6
        if len(prompt) == 331: return 7
        if len(prompt) == 341: return 8
        if len(prompt) == 399: return 9
        if len(prompt) == 471: return 10
    elif "news" in prompt:
        if len(prompt) == 30: return 1
        if len(prompt) == 60: return 2
        if len(prompt) == 131: return 3
        if len(prompt) == 206: return 4
        if len(prompt) == 272: return 5
        if len(prompt) == 356: return 6
        if len(prompt) == 442: return 7
        if len(prompt) == 514: return 8
        if len(prompt) == 566: return 9
        if len(prompt) == 664: return 10