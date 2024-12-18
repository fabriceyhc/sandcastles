import os
import pandas as pd

# Directory containing attack trace files
attack_trace_dir = "./distinguisher/attack_traces/"

# Helper function to extract variables from filenames
def parse_filename(filename):
    print(filename)
    # Example filename: "{o_str}_{w_str}_{m_str}_compare-original={compare_against_original}_attack_results.csv"
    # Strip the directory and extension, then split by '_'
    base_name = os.path.basename(filename).replace('_attack_results.csv', '')
    parts = base_name.split('_')
    
    o_str = parts[0]
    w_str = parts[1]
    m_str = parts[2]
    compare_against_original = parts[3].split('=')[1]
    
    return o_str, w_str, m_str, compare_against_original

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

# Parse the directory and process each attack trace file
def process_attack_traces(directory):
    all_attacks = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract variables from the filename
            o_str, w_str, m_str, compare_against_original = parse_filename(filename)
            
            # Load the CSV into a pandas DataFrame
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Separate attacks based on step_num
            attacks = separate_attacks(df)
            
            # Store each attack with its metadata
            for i, attack in enumerate(attacks):
                all_attacks.append({
                    'o_str': o_str,
                    'w_str': w_str,
                    'm_str': m_str,
                    'compare_against_original': compare_against_original,
                    'attack_num': i + 1,
                    'attack_data': attack
                })
    
    return all_attacks

# Call the function and store the parsed attacks
all_attacks = process_attack_traces(attack_trace_dir)