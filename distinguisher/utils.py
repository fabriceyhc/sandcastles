import os
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

# Directory containing attack trace files
attack_trace_dir = "attack/traces/"

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

def process_attack_traces(directory):
    all_attacks = []
    file_groups = {}
    
    for filename in os.listdir(directory):
        if "annotated" in filename or "DocumentMutator" in filename:
            continue
        if 'results' in filename:
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

# Call the function and store the parsed attacks
all_attacks = process_attack_traces(attack_trace_dir)