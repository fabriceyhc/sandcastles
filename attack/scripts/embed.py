import os
import pandas as pd
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer
from glob import glob
from tqdm import tqdm
import warnings

# Suppress pandas warnings about appending to DataFrame
warnings.filterwarnings('ignore')

# Configuration
WATERMARK_TYPES = ["Adaptive", "KGW", "SIR"]
MUTATORS = [
    "DocumentMutator",
    "Document1StepMutator",
    "Document2StepMutator",
    "SentenceMutator",
    "SpanMutator",
    "EntropyWordMutator",
    "WordMutator",
]

def get_save_paths(input_dir, csv_filename):
    """Generate save paths in the embeddings directory"""
    base_name = os.path.splitext(csv_filename)[0]
    
    return {
        'embeddings': os.path.join(EMBEDDINGS_DIR, f"{base_name}_embeddings.h5"),
        'data': os.path.join(EMBEDDINGS_DIR, f"{base_name}_filtered.parquet"),
        'embeddings_dir': EMBEDDINGS_DIR
    }

def extract_components_from_filename(filename):
    """Extract watermark type and mutator from filename"""
    filename_lower = filename.lower()
    
    # Detect watermark type
    watermark_type = None
    for wt in WATERMARK_TYPES:
        if wt.lower() in filename_lower:
            watermark_type = wt
            break
            
    # Detect mutator
    mutator = None
    for m in MUTATORS:
        if m.lower() in filename_lower:
            mutator = m
            break
            
    return watermark_type, mutator

def process_csv(csv_path, model, force_recompute=False):
    """Process a single CSV file"""
    csv_filename = os.path.basename(csv_path)
    save_paths = get_save_paths(os.path.dirname(csv_path), csv_filename)
    
    # Create embeddings directory if needed
    os.makedirs(save_paths['embeddings_dir'], exist_ok=True)
    
    # Check if we can skip processing
    if not force_recompute and all(os.path.exists(p) for p in [save_paths['embeddings'], save_paths['data']]):
        return None  # Already processed
    
    try:
        # Load and process CSV
        df = pd.read_csv(csv_path)
        
        # Apply OR filter
        filtered = df[
            (df['step_num'] == -1) | 
            (df['quality_preserved'] == True)
        ].copy()
        
        if filtered.empty:
            return None  # No valid rows
        
        # Add source file information
        filtered['source_file'] = csv_filename
        
        # Compute embeddings
        texts = filtered['current_text'].tolist()
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Save results
        with h5py.File(save_paths['embeddings'], 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings)
        
        filtered.to_parquet(save_paths['data'])
        
        return len(filtered)
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None

def process_attack_traces(input_dir, force_recompute=False):
    """Process all CSVs in directory with per-file embedding storage"""
    csv_files = glob(os.path.join(input_dir, "*.csv"))
    csv_files = [p for p in csv_files if "Word" in p]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Processing {len(csv_files)} CSV files...")
    processed = 0
    with tqdm(total=len(csv_files)) as pbar:
        for csv_path in csv_files:
            result = process_csv(csv_path, model, force_recompute)
            if result is not None:
                processed += 1
            pbar.update(1)
    
    print(f"\nProcessed {processed} files (skipped {len(csv_files)-processed} files)")

def load_embeddings_dict(embeddings_dir):
    """Load embeddings organized in dictionary by (watermark_type, mutator)"""
    embedding_dict = {}
    
    # Find all embedding files
    emb_files = glob(os.path.join(embeddings_dir, "*_embeddings.h5"))
    
    for emb_file in emb_files:
        # Get corresponding parquet file
        base_name = emb_file.replace("_embeddings.h5", "")
        parquet_file = f"{base_name}_filtered.parquet"
        
        if not os.path.exists(parquet_file):
            continue
            
        # Extract components from filename
        filename = os.path.basename(emb_file)
        watermark_type, mutator = extract_components_from_filename(filename)
        
        if not watermark_type or not mutator:
            continue
            
        # Load data and embeddings
        data = pd.read_parquet(parquet_file)
        with h5py.File(emb_file, 'r') as hf:
            embeddings = hf['embeddings'][:]
            
        # Create or update dictionary entry
        key = (watermark_type, mutator)
        if key not in embedding_dict:
            embedding_dict[key] = {
                'embeddings': [],
                'data': []
            }
            
        embedding_dict[key]['embeddings'].append(embeddings)
        embedding_dict[key]['data'].append(data)
    
    # Concatenate arrays and DataFrames for each key
    for key in embedding_dict:
        embedding_dict[key]['embeddings'] = np.concatenate(embedding_dict[key]['embeddings'])
        embedding_dict[key]['data'] = pd.concat(embedding_dict[key]['data'], ignore_index=True)
    
    sorted_dict = dict(sorted(embedding_dict.items(), key=lambda x: (x[0][0], x[0][1])))
    
    return sorted_dict

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "./attack/traces/" 
    EMBEDDINGS_DIR = "./attack/traces/embeddings/"
    
    # Step 1: Process all CSVs (only needs to run once)
    process_attack_traces(INPUT_DIR, force_recompute=False)
    process_attack_traces(f"{INPUT_DIR}/annotated", force_recompute=False)
    
    # Step 2: Load organized embeddings
    embedding_dict = load_embeddings_dict(EMBEDDINGS_DIR)
    
    # Print summary
    print("\nLoaded embeddings summary:")
    for (wm_type, mutator), data in embedding_dict.items():
        print(f"{wm_type: <9} + {mutator: <20}: {data['embeddings'].shape[0]: >6} embeddings")
    
    # Example access:
    if ('KGW', 'WordMutator') in embedding_dict:
        kgw_data = embedding_dict[('KGW', 'WordMutator')]['data']
        print("\nExample KGW WordMutator data:")
        print(kgw_data.head())