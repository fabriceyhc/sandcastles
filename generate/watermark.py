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
        assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'UPV', 'EXP', 'EXPEdit', 'EXPGumbel', 'SynthID']
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
        best_result = {
            "prompt": prompt,
            "text": "",
            "is_watermarked": False,
            "score": -float("inf"),
            "time": 0,
            "num_attempts": 0,
            "algorithm": self.algorithm_name,
        }

        for attempt in range(retries):
            print(f"[{self.algorithm_name}] Attempt {attempt + 1}/{retries} for prompt: {prompt[:30]}...")
            start_time = time.time()

            # Generate watermarked text
            generated_text = self.watermark.generate_watermarked_text(prompt)

            # Remove prompt from generated text if it exists
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            detection_results = self.watermark.detect_watermark(generated_text)
            is_watermarked = detection_results.get('is_watermarked', False)
            score = detection_results.get('score', 0)
            elapsed_time = time.time() - start_time
            print(f"[{self.algorithm_name}] Generated text with 'is_watermarked': {is_watermarked}, score: {score:.2f} in {elapsed_time:.2f}s")

            # Update best result
            if score > best_result["score"]:
                best_result.update({
                    "text": generated_text,
                    "is_watermarked": is_watermarked,
                    "score": score,
                    "time": elapsed_time,
                    "num_attempts": attempt + 1,
                })

            # If watermark is successfully detected, stop retrying
            if is_watermarked:
                print(f"[{self.algorithm_name}] Watermark successfully detected. Stopping retries.")
                break

        return best_result

    def unwatermarked_prompt(self, prompt, retries=5):
        best_result = {
            "prompt": prompt,
            "text": "",
            "is_watermarked": True,  # Assume worst case initially
            "score": float("inf"),
            "scores": {},
            "time": 0,
            "num_attempts": 0,
        }

        for attempt in range(retries):
            print(f"[Unwatermarked] Attempt {attempt + 1}/{retries} for prompt: {prompt[:30]}...")
            start_time = time.time()
            try:
                # Generate unwatermarked text
                generated_text = self.watermark.generate_unwatermarked_text(prompt)

                # Remove prompt from generated text if it exists
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

                detection_results = self.watermark.detect_watermark(generated_text)
                is_watermarked = detection_results.get('is_watermarked', False)
                score = detection_results.get('score', 0)
                elapsed_time = time.time() - start_time
                print(f"[Unwatermarked] Generated text with 'is_watermarked': {is_watermarked}, score: {score:.2f} in {elapsed_time:.2f}s")

                # Update best result
                if score < best_result["score"]:
                    best_result.update({
                        "text": generated_text,
                        "is_watermarked": is_watermarked,
                        "score": score,
                        "time": elapsed_time,
                        "num_attempts": attempt + 1,
                    })

                # If watermark is not detected, stop retrying
                if not is_watermarked:
                    print(f"[Unwatermarked] Watermark not detected. Stopping retries.")
                    break

            except Exception as e:
                print(f"[Unwatermarked] Error during generation: {e}")
                continue

        return best_result

    def watermark_dataset(self, dataframe, output_csv, n_responses=1, retries=5):
        existing_responses = load_existing_responses(output_csv)

        for _, row in dataframe.iterrows():
            prompt = row['prompt']

            # Check if the prompt already has the required number of responses
            if existing_responses.get(prompt, 0) >= n_responses:
                print(f"[{self.algorithm_name}] Skipping prompt: {prompt[:30]} (has {existing_responses[prompt]} responses, needs {n_responses}).")
                continue

            # Generate additional responses if needed
            responses_needed = n_responses - existing_responses.get(prompt, 0)
            print(f"[{self.algorithm_name}] Generating {responses_needed} response(s) for prompt: {prompt[:30]}...")
            for _ in range(responses_needed):
                result = self.watermark_prompt(prompt, retries=retries)

                # Include the original row data with the result
                combined_result = {**row.to_dict(), **result}
                save_to_csv([combined_result], output_csv)

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m generate.watermark

    # Processing log
    print("Starting watermark generation...")

    # Parameters
    model_id = "meta-llama/Llama-3.1-70B-Instruct"
    data_path = "./data/prompts/entropy_control.csv"

    # Load data
    df = pd.read_csv(data_path) #.head(5)

    # List of watermarking algorithms
    algorithms = ["KGW", "EXP"] # "SynthID", "EXP", "EXPGumbel", 

    # Process and save results incrementally
    n_responses_per_prompt = 3
    for i, algorithm_name in enumerate(algorithms):
        print(f"\nProcessing algorithm: {algorithm_name}")
        output_csv = f"./data/texts/entropy_control_{algorithm_name}.csv"
        processor = WatermarkProcessor(algorithm_name, model_id)
        processor.watermark_dataset(df, output_csv, n_responses=n_responses_per_prompt, retries=3)

    # Generate unwatermarked samples
    print("\nGenerating unwatermarked samples...")
    unwatermarked_output_csv = "./data/texts/entropy_control_unwatermarked.csv"
    all_scores = {}
    for algorithm_name in algorithms:
        processor = WatermarkProcessor(algorithm_name, model_id)
        for _, row in df.iterrows():
            prompt = row['prompt']
            unwatermarked_result = processor.unwatermarked_prompt(prompt, retries=3)
            all_scores.update(unwatermarked_result.pop('scores', {}))
            combined_result = {**row.to_dict(), **unwatermarked_result}
            save_to_csv([combined_result], unwatermarked_output_csv)