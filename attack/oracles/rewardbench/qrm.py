import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from attack.oracles.base import ResponseQuality
from attack.oracles.utils import add_prefix_to_keys

from attack.oracles.rewardbench._base import BaseRewardBenchOracle

class QRMOracle(BaseRewardBenchOracle):
    """
    An oracle class that uses QRM-Gemma-2-27B or Llama3.1-8B for reward scoring.
    This class outputs a reward distribution over 5 attributes: helpfulness, 
    correctness, coherence, complexity, and verbosity and we average over all. 
    """

    def __init__(
        self,
        model_path: str = "nicolinho/QRM-Gemma-2-27B", # "nicolinho/QRM-Llama3.1-8B-v2"
        device: str = "cuda",
        similarity_threshold: float = 0.4615293560606061, # TODO: find this number
        use_flash_attn: bool = True,
        **model_kwargs
    ):
        """
        :param model_path: HF-style path or local directory for QRM-Gemma-2-27B
        :param device: 'cuda', 'cpu', or 'auto' to control device placement
        :param similarity_threshold: threshold for deciding TIE vs. A_BETTER / B_BETTER
        :param use_flash_attn: whether to enable 'flash_attention_2'
        :param model_kwargs: additional arguments to pass into from_pretrained
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        # For large models, consider device_map="auto" if you need multi-GPU inference.
        attn_implementation = "flash_attention_2" if use_flash_attn else None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir="/data2/.shared_models/",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=device,
            trust_remote_code=True,  # QRM-Gemma uses custom code
            **model_kwargs
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        self.similarity_threshold = similarity_threshold
        print(f"QRMOracle({model_path}) loaded to {self.device}")


    def _score_example(self, prompt, text):
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text}
        ]
        # Tokenize using the provided `apply_chat_template` method if available
        chat_tokenized = self.tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors="pt"
        ).to(self.device)
        # Forward pass for each chat
        with torch.no_grad():
            output = self.model(chat_tokenized)
        score = output.score.mean(dim=1).cpu().item()
        return score

    def evaluate(self, instruction, response_A, response_B, explain=False, **kwargs):
        """
        Return model-based 'scores' for two candidate responses to the same instruction.
        
        By design, QRM-Gemma-2-27B outputs a vector of size 5 (the five reward attributes).
        We reduce it to a single scalar per response (by averaging or summing) so we can 
        label which response is "better" or if it is a "tie" for compatibility with 
        the rest of the interface.

        :param instruction: The prompt or user query
        :param response_A: Candidate response A
        :param response_B: Candidate response B
        :param explain: (unused) for interface parity
        :param kwargs: Extra arguments (ignored but kept for interface consistency)
        :return: Dict with "score_A" and "score_B"
        """
        return {
            "score_A": self._score_example(instruction, response_A),
            "score_B": self._score_example(instruction, response_B),
        }

    def extract_label(self, evaluation):
        """
        Given a dict of the form {"score_A": float, "score_B": float}, decide if:
          - abs(score_A - score_B) <= similarity_threshold => TIE
          - score_A > score_B => A_BETTER
          - else => B_BETTER
        """
        score_A = evaluation["score_A"]
        score_B = evaluation["score_B"]
        if abs(score_A - score_B) <= self.similarity_threshold:
            return ResponseQuality.TIE
        elif score_A > score_B:
            return ResponseQuality.A_BETTER
        else:
            return ResponseQuality.B_BETTER

    def is_quality_preserved(self, instruction, original_text, mutated_text, **kwargs):
        """
        Compare the order of preference for (original, mutated) vs. (mutated, original).
        If the preference flips in a certain way, we consider the original "quality" to 
        be preserved or not.

        :return: dict with both evaluations and a "quality_preserved" boolean.
        """
        original = self.evaluate(
            instruction=instruction,
            response_A=original_text,
            response_B=mutated_text,
            **kwargs
        )
        followup = self.evaluate(
            instruction=instruction,
            response_A=mutated_text,
            response_B=original_text,
            **kwargs
        )  # reversed outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # Based on the logic in your sample code:
        # if original says B_BETTER or TIE, and followup says A_BETTER or TIE,
        # we interpret that as "quality is preserved" under the mutation.
        if original_pred in [ResponseQuality.B_BETTER, ResponseQuality.TIE] \
           and followup_pred in [ResponseQuality.A_BETTER, ResponseQuality.TIE]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original_prefixed = add_prefix_to_keys(original, "original_")
        followup_prefixed = add_prefix_to_keys(followup, "followup_")
        original_prefixed.update(followup_prefixed)
        original_prefixed.update({"quality_preserved": is_quality_preserved})
        return original_prefixed

    def test(self, instruction, response_A, response_B, label, **kwargs):
        """
        Evaluate the model's performance on a known-labeled pair. 
        - 'label' is an instance of ResponseQuality.{A_BETTER, B_BETTER, TIE}.
        - We check if the model agrees with 'label' and whether the reversed order 
          also agrees with the 'inverted' label, to measure positional bias.

        :return: structured dict containing predictions, correctness, etc.
        """
        original_label = label
        followup_label = self.invert_label(label)

        original = self.evaluate(instruction, response_A, response_B, **kwargs)
        followup = self.evaluate(instruction, response_B, response_A, **kwargs)

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # Score correctness
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1  # Both correct
        elif (original_label == original_pred) or (followup_label == followup_pred):
            pred_correct = 0.5  # One correct, one mismatch => partial credit

        # Merge into a single result dict
        original_prefixed = add_prefix_to_keys(original, "original_")
        followup_prefixed = add_prefix_to_keys(followup, "followup_")
        original_prefixed.update({
            **followup_prefixed,
            "original_label": original_label,
            "followup_label": followup_label,
            "original_pred": original_pred,
            "followup_pred": followup_pred,
            "pred_correct": pred_correct,
        })
        return original_prefixed

    @staticmethod
    def invert_label(label):
        """
        Utility to invert A_BETTER <-> B_BETTER, leaving TIE unchanged.
        """
        if label == ResponseQuality.A_BETTER:
            return ResponseQuality.B_BETTER
        elif label == ResponseQuality.B_BETTER:
            return ResponseQuality.A_BETTER
        return label



# Testing
if __name__ == "__main__":

    import time
    
    def test():      

        # Load sample data row

        instruction = "In what country is Berlin?"
        original_text = "Berlin is in Germany."
        mutated_text = "Berlin is in Hungary."
        label = ResponseQuality.A_BETTER

        oracle = QRMGemmaOracle()

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