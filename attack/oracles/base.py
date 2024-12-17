from abc import ABC, abstractmethod
from functools import partial
from enum import Enum
from attack.oracles.utils import add_prefix_to_keys
from guidance import models 

class ResponseQuality(Enum):
    A_BETTER = 1
    B_BETTER = 2
    TIE = 3

# Abstract base class for all oracles
class Oracle(ABC):

    def __init__(self, llm=None, explain=False) -> None:
        self.llm = llm
        if self.llm is None:
            self.llm = self._initialize_llm("/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf")
        self.explain=explain

    def _initialize_llm(self, path):
        llm = models.LlamaCpp(
            model=path,
            echo=False,
            n_gpu_layers=-1,
            n_ctx=4096
        )
        return llm

    @property
    @abstractmethod
    def input_keys(self):
        pass

    @property
    @abstractmethod
    def output_keys(self):
        pass

    @staticmethod
    @abstractmethod
    def annotation_fn(lm, explain=False, **kwargs):
        pass

    @staticmethod
    def apply_annotation(input_dict, llm, annotation_fn, input_keys, output_keys, persona=None, prefix="", **kwargs):
        inputs = {k: input_dict[k] for k in input_keys}
        output = llm+annotation_fn(persona=persona, **inputs, **kwargs)
        return {prefix+k: output[k] for k in output_keys}

    def annotate(self, input_dict, **kwargs):
        return self.apply_annotation(
            input_dict=input_dict, 
            llm=self.llm, 
            annotation_fn=partial(self.annotation_fn, explain=self.explain),
            input_keys=self.input_keys, 
            output_keys=self.output_keys,
            **kwargs
        )

    def annotate_dataset(self, dataset, prefix=""):
        return dataset.map(
            partial(self.apply_annotation, 
                llm=self.llm, 
                annotation_fn=partial(self.annotation_fn, explain=self.explain),
                input_keys=self.input_keys, 
                output_keys=self.output_keys, 
                prefix=prefix
        )
    )

    @abstractmethod
    def extract_label(self, evaluation):
        pass

    def evaluate(self, instruction, response_A, response_B, **kwargs):
        input_dict = {
            "instruction": instruction, 
            "response_A": response_A,
            "response_B": response_B
        }
        evaluation = self.annotate(input_dict, **kwargs)
        return evaluation

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
        #elif (original_label == original_pred) or (followup_label == followup_pred):
        #    pred_correct = 0.5 # one was correct, but some positional bias was present

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
    