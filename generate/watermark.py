import os
import time
import pandas as pd
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Simple save_to_csv function
def save_to_csv(data, filepath, rewrite=False):
    mode = 'w' if rewrite else 'a'
    header = rewrite or not os.path.exists(filepath)
    pd.DataFrame(data).to_csv(filepath, mode=mode, header=header, index=False)

def load_existing_responses(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        existing_data = pd.read_csv(filepath)
        if 'prompt' not in existing_data.columns:
            return {}
        return existing_data.groupby('prompt').size().to_dict()
    except Exception as e:
        print(f"Could not load existing responses from {filepath}. Error: {e}")
        return {}

class WatermarkProcessor:
    model = None
    tokenizer = None

    @classmethod
    def initialize_model_and_tokenizer(cls, model_id):
        if cls.model is None or cls.tokenizer is None:
            cls.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map='auto', 
                cache_dir="/data2/.shared_models"
            )
            cls.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True, 
                cache_dir="/data2/.shared_models"
            )
            cls.tokenizer.pad_token = cls.tokenizer.pad_token or cls.tokenizer.eos_token

    def __init__(self, algorithm_name, model_id, config_dir="./generate/config"):
        assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'UPV', 'EXP', 'EXPEdit', 'SynthID'] + ['Unwatermarked']
        self.algorithm_name = algorithm_name

        # Ensure model and tokenizer are initialized once
        self.initialize_model_and_tokenizer(model_id)

        # Transformers configuration
        self.transformers_config = TransformersConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_size=self.tokenizer.vocab_size,
            max_new_tokens=1024,
            min_length=128,
            do_sample=True,
            no_repeat_ngram_size=0,
        )

        # Load watermark algorithm
        self.watermark = AutoWatermark.load(
            algorithm_name, 
            algorithm_config=f'{config_dir}/{algorithm_name}.json', 
            transformers_config=self.transformers_config
        )

    def watermark_prompt(self, prompt, retries=5):
        best_score = -float("inf")
        best_text = ""
        best_time = 0

        for attempt in range(retries):
            start_time = time.time()
            try:
                # Generate watermarked text
                if "unwatermarked" in self.algorithm_name.lower():
                    generated_text = self.watermark.generate_unwatermarked_text(prompt)
                else:
                    generated_text = self.watermark.generate_watermarked_text(prompt)
                detection_results = self.watermark.detect_watermark(generated_text)

                # Update best result if score improves
                score = detection_results.get('score', 0)
                if score > best_score:
                    best_score = score
                    best_text = generated_text
                    best_time = time.time() - start_time

                # Break early if desired score is achieved
                if best_score >= 3:
                    break

            except Exception as e:
                print(f"Error during watermarking for prompt: {prompt}. Error: {e}")
                continue

        return {
            "prompt": prompt,
            "watermarked_text": best_text,
            "score": best_score,
            "time": best_time,
            "algorithm": self.algorithm_name,
        }

    def watermark_dataset(self, dataframe, output_csv, n_responses=1, retries=5):
        existing_responses = load_existing_responses(output_csv)

        for _, row in dataframe.iterrows():
            prompt = row['prompt']

            # Check if the prompt already has the required number of responses
            if existing_responses.get(prompt, 0) >= n_responses:
                print(f"Skipping prompt: {prompt} (already has {existing_responses[prompt]} responses)")
                continue

            # Generate additional responses if needed
            responses_needed = n_responses - existing_responses.get(prompt, 0)
            for _ in range(responses_needed):
                result = self.watermark_prompt(prompt, retries=retries)

                # Include the original row data with the result
                combined_result = {**row.to_dict(), **result}

                # Save result incrementally
                save_to_csv([combined_result], output_csv)

# Parameters
model_id = "meta-llama/Llama-3.3-70B-Instruct"
data_path = "./data/prompts/entropy_control.csv"

# Load data
df = pd.read_csv(data_path).sample(n=2)

# List of watermarking algorithms
algorithms = ["KGW", "SynthID", "Unwatermarked"]

# Process and save results incrementally
n_responses_per_prompt = 3
for algorithm_name in algorithms:
    output_csv = f"./data/texts/entropy_control_{algorithm_name}.csv"
    processor = WatermarkProcessor(algorithm_name, model_id)
    processor.watermark_dataset(df, output_csv, n_responses=n_responses_per_prompt, retries=3)
