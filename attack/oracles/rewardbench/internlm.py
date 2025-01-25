# RUN: CUDA_VISIBLE_DEVICES=0,1 python -m oracles.rewardbench.internlm

import torch
from transformers import AutoModel, AutoTokenizer
from attack.oracles.base import ResponseQuality
from attack.oracles.utils import add_prefix_to_keys
from attack.oracles.rewardbench._base import BaseRewardBenchOracle

class InternLMOracle(BaseRewardBenchOracle):
    
    def __init__(self, model=None, explain=False) -> None:
        self.device = torch.device("cuda"if torch.cuda.is_available() else"cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-20b-reward", trust_remote_code=True)
        if model is None:
            self.model = AutoModel.from_pretrained(
                "internlm/internlm2-20b-reward", 
                cache_dir="/data2/.shared_models/",
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                device_map="auto"
            )
        print(f"InternLMOracle(internlm/internlm2-20b-reward) loaded to {self.device}")
        self.similarity_threshold = 0.4615293560606061

    def _score_example(self, prompt, text):
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text}
        ]
        score = self.model.get_score(self.tokenizer, chat)
        return score 

    def evaluate(self, instruction, response_A, response_B, explain=False, **kwargs):
        chat_A = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_A}
        ]
        chat_B = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_B}
        ]
        
        # Get scores for both chats
        score_A, score_B = self.model.get_scores(self.tokenizer, [chat_A, chat_B])
        
        # Return output
        return {
            "score_A": score_A,
            "score_B": score_B,
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

        # Load sample data row

        instruction = "In what country is Berlin?"
        original_text = "Berlin is in Germany."
        mutated_text = "Berlin is in Hungary."
        label = ResponseQuality.A_BETTER

        oracle = InternLMOracle()

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