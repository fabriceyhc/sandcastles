from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import LlamaPreTrainedModel, LlamaModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from attack.oracles.base import ResponseQuality
from attack.oracles.rewardbench._base import BaseRewardBenchOracle

class INFORMForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class INFORMOracle(BaseRewardBenchOracle):

    def __init__(
        self,
        model_name: str = "infly/INF-ORM-Llama3.1-70B",
        explain: bool = False,
        similarity_threshold: float = 0.4615293560606061, # TODO: find this number
        **model_kwargs
    ) -> None:
        """
        :param model_name: HuggingFace-style model name or local path
        :param explain: Whether to enable any 'explain' functionality (not used here)
        :param similarity_threshold: Score difference threshold for tie vs. better label
        :param model_kwargs: Additional kwargs to pass to from_pretrained (e.g., device_map, torch_dtype, etc.)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        
        # Load INFORM model
        # Make sure that INFORMForSequenceClassification is visible to this module or import path is correct
        self.model = INFORMForSequenceClassification.from_pretrained(
            model_name,
            cache_dir="/data2/.shared_models/",
            torch_dtype=torch.bfloat16,       # or torch.float16, depending on your GPU
            device_map="auto",                # or "cuda" if you want to specify a single device
            attn_implementation="flash_attention_2",
            num_labels=1,
            **model_kwargs
        )
        self.similarity_threshold = similarity_threshold
        print(f"INFORMOracle({model_name}) loaded to {self.device}")


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
        score = output.logits[0].item()
        return score

    def evaluate(self, instruction, response_A, response_B, explain=False, **kwargs):
        """
        Return model-based scores for two candidate responses to the same instruction.
        
        :param instruction: The prompt or user query
        :param response_A: Candidate response A
        :param response_B: Candidate response B
        :param explain: (Unused) For parity with the interface
        :param kwargs: Extra arguments (ignored here but kept for interface compatibility)
        :return: Dict with 'score_A' and 'score_B'
        """
        return {
            "score_A": self._score_example(instruction, response_A),
            "score_B": self._score_example(instruction, response_B),
        }

    def extract_label(self, evaluation):
        """
        Given the dict from self.evaluate(...), return an enum describing which response is better.
        If the absolute difference is below similarity_threshold, treat it as a tie.
        """
        score_A, score_B = evaluation["score_A"], evaluation["score_B"]
        if abs(score_A - score_B) <= self.similarity_threshold:
            return ResponseQuality.TIE
        elif score_A > score_B:
            return ResponseQuality.A_BETTER
        else:
            return ResponseQuality.B_BETTER

    def is_quality_preserved(self, instruction, original_text, mutated_text, **kwargs):
        """
        Check whether the 'quality' ordering is preserved when swapping original vs mutated text.
        Returns a dictionary with both evaluations and a boolean for 'quality_preserved'.
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
        )  # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # If original says B is better or tie, and followup says A is better or tie,
        # we interpret that as "quality was preserved" under the mutation.
        if original_pred in [ResponseQuality.B_BETTER, ResponseQuality.TIE] \
           and followup_pred in [ResponseQuality.A_BETTER, ResponseQuality.TIE]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update(followup)
        original.update({"quality_preserved": is_quality_preserved})
        return original

    def test(self, instruction, response_A, response_B, label, **kwargs):
        """
        Evaluate the model on a known-labeled pair, checking correctness 
        when the order of A/B is switched. This helps measure model's position bias.
        """
        original_label = label
        followup_label = self.invert_label(label)

        original = self.evaluate(instruction, response_A, response_B, **kwargs) 
        followup = self.evaluate(instruction, response_B, response_A, **kwargs)  # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # Assign correctness points
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1  # Both are correct and positionally invariant
        elif (original_label == original_pred) or (followup_label == followup_pred):
            pred_correct = 0.5  # One was correct, but there's some positional bias

        # Prepare structured output
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
        """
        Utility for flipping A_BETTER <-> B_BETTER while leaving TIE unchanged.
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

        instruction = "In what country is Berlin?"
        original_text = "Berlin is in Germany."
        mutated_text = "Berlin is in Hungary."
        label = ResponseQuality.A_BETTER

        oracle = INFORMOracle()

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