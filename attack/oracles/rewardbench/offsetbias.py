# RUN: CUDA_VISIBLE_DEVICES=7 python -m oracles.rewardbench.offsetbias

import torch
from transformers import AutoModel, AutoTokenizer, pipeline
from attack.oracles.base import ResponseQuality
from attack.oracles.utils import add_prefix_to_keys
from attack.oracles.rewardbench._base import BaseRewardBenchOracle

from typing import Dict, List
import torch


class OffsetBiasPipeline:
    def __init__(self, 
                 model_id="NCSOFT/Llama-3-OffsetBias-RM-8B", 
                 device_map="auto", 
                 torch_dtype=torch.bfloat16, 
                 truncation=True, 
                 trust_remote_code=False, 
                 max_length=4096
        ):
        self.truncation = truncation
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_id,
            tokenizer=self.tokenizer,
            model_kwargs={
                "torch_dtype": torch.bfloat16, 
                "cache_dir": "/data2/.shared_models/",
                "device_map": "auto"
            }
        )

        self.pipe_kwargs = {
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 1
        }

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=False
        ).replace(self.tokenizer.bos_token, "")
        with torch.no_grad():
            score = self.pipe(input_texts, **self.pipe_kwargs)[0]["score"]
        return score

class OffsetBiasOracle(BaseRewardBenchOracle):
    
    def __init__(self, model=None, explain=False) -> None:
        if model is None:
            self.model = OffsetBiasPipeline()
        self.similarity_threshold = 0.7130681818181819

    def _score_example(self, prompt, text):
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text}
        ]
        score = self.model(chat)
        return score

    def evaluate(self, instruction, response_A, response_B, explain=False, **kwargs):
        return {
            "score_A": self._score_example(instruction, response_A),
            "score_B": self._score_example(instruction, response_B),
        }
    
    
    def extract_label(self, evaluation):
        score_A, score_B = evaluation["score_A"], evaluation["score_B"]
        if abs(score_A - score_B) <= self.similarity_threshold:
            return ResponseQuality.TIE
        elif score_A > score_B:
            return ResponseQuality.A_BETTER
        else:
            return ResponseQuality.B_BETTER

    def is_quality_preserved(self, instruction, original_text, mutated_text, **kwargs):
        
        original = self.evaluate(instruction, response_A=original_text, response_B=mutated_text, **kwargs) 
        followup = self.evaluate(instruction, response_A=mutated_text, response_B=original_text, **kwargs) # switched outputs
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [ResponseQuality.B_BETTER, ResponseQuality.TIE] and followup_pred in [ResponseQuality.A_BETTER, ResponseQuality.TIE]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({**followup})
        original.update({"quality_preserved": is_quality_preserved})
        return original

    def test(self, instruction, response_A, response_B, label, **kwargs):
        original_label = label
        followup_label = self.invert_label(label)

        original = self.evaluate(instruction, response_A, response_B, **kwargs) 
        followup = self.evaluate(instruction, response_B, response_A, **kwargs) # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1 # both are correct and positionally invariant
        elif (original_label == original_pred) or (followup_label == followup_pred):
            pred_correct = 0.5 # one was correct, but some positional bias was present

        # prepare output
        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({
            **followup,
            "original_label": original_label,
            "followup_label": followup_label,
            "original_pred": original_pred, 
            "followup_pred": followup_pred,
            "pred_correct": pred_correct,
        })

        return original

    @staticmethod
    def invert_label(label):
        if label == ResponseQuality.A_BETTER:
            return ResponseQuality.B_BETTER
        elif label == ResponseQuality.B_BETTER:
            return ResponseQuality.A_BETTER
        return label # TIE stays the same


# Testing
if __name__ == "__main__":

    import time

    def test():

        instruction = "In what country is Berlin?"
        original_text = "Berlin is in Germany."
        mutated_text = "Berlin is in Hungary."
        label = ResponseQuality.A_BETTER

        oracle = OffsetBiasOracle()

        # Run quality assessments
        start = time.time()
        quality_eval = oracle.is_quality_preserved(
            instruction=instruction, 
            original_text=original_text, 
            mutated_text=mutated_text, 
            reference_answer=None
        )
        delta = time.time() - start
        print("EVAL oracle.is_quality_preserved")
        print("quality_eval:", quality_eval)
        print("time_taken:", delta)

        print("EVAL  oracle.test:")
        start = time.time()
        results = oracle.test(instruction, original_text, mutated_text, label)
        delta = time.time() - start
        print(results)
        print("time_taken:", delta)
        

    test()
    