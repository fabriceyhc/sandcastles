# RUN: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m oracles.guidance.mutation

import guidance
from guidance import gen, select, user, system, assistant
from attack.oracles.base import Oracle, ResponseQuality
from attack.oracles.utils import add_prefix_to_keys

class MutationOracle(Oracle):

    @property
    def input_keys(self):
        return ["instruction", "response_A", "response_B",]

    @property
    def output_keys(self):
        return ["answer", "explanation"] if self.explain else ["answer"]

    @staticmethod
    @guidance
    def annotation_fn(lm, explain=False, **kwargs):
        pattern = 'Yes|No'
        if kwargs["persona"] is not None:
          with system():
              lm += f"{kwargs['persona']}"
                
        if kwargs["original_first"]:
          with user():
              lm += f"""\
              ### Instructions: 
              We are seeking your help to find an answer to this problem:
              The following is a prompt that was given to an AI assistant, and its corresponding response.
              After that, is the same answer, but rephrased. 

              ### Here is the prompt:
              {kwargs['instruction']}

              ### Original Response:
              {kwargs['response_A']}

              ### Rephrased Response:
              {kwargs['response_B']}

              ### Instructions: 
              We want to know if the rephrased answer maintains the same level of quality and accuracy as the original.
              Please make your evaluation based on the level of grammatical correctness, fluency, accuracy, structure, and clarity in the original vs rephrased answers.
              Be strict in your evaluation and consider the overall quality of the response, and take note of the differences between the two.
              If the rephrased response is just as good or better, output "Yes", otherwise output "No".
              """
        else:
          with user():
              lm += f"""\
              ### Instructions: 

              We are seeking your help to find an answer to this problem:
              The following is a prompt given to an AI assistant, and a rephrased version of the response. 
              After that, is the original answer. 
              
              ### Here is the prompt:
              {kwargs['instruction']}

              ### Rephrased Response:
              {kwargs['response_B']}
              
              ### Original Response:
              {kwargs['response_A']}

              ### Instructions: 
              We want to know if the rephrased answer maintains the same level of quality and accuracy as the original.
              Please make your evaluation based on the level of grammatical correctness, fluency, accuracy, structure, and clarity in the original vs rephrased answers.
              Be strict in your evaluation and consider the overall quality of the response, and take note of the differences between the two.
              If the rephrased response is just as good or better, output "Yes", otherwise output "No".
              """
        with assistant():
            if explain:
              lm += f"""\
                ### Brief Explanation (100 words max):
                {gen(name='explanation', max_tokens=200, stop=["<|eot_id|>"])}
                """
            lm += f"""{gen(regex=pattern, name='answer')}"""
        return lm

    def extract_label(self, evaluation):
        goal = evaluation["answer"].lower() == "yes"
        if goal:
            label = ResponseQuality.B_BETTER
        else:
            label = ResponseQuality.A_BETTER
        return label
    
    def is_quality_preserved(self, instruction, original_text, mutated_text, **kwargs):
        kwargs['original_first'] = True
        original = self.evaluate(instruction, response_A=original_text, response_B=mutated_text, **kwargs) 
        kwargs['original_first'] = False
        followup = self.evaluate(instruction, response_A=original_text, response_B=mutated_text, **kwargs) # switched outputs
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)
        
        if original_pred in [ResponseQuality.B_BETTER] and followup_pred in [ResponseQuality.A_BETTER]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({**followup})
        original.update({"quality_preserved": is_quality_preserved})
        return original
    
    def test(self, instruction, response_A, response_B, label, **kwargs):
        if label == ResponseQuality.TIE:
            label = ResponseQuality.B_BETTER
        original_label = label
        followup_label = label

        kwargs.update({'original_first' : True})
        original = self.evaluate(instruction, response_A, response_B, **kwargs) 
        kwargs.update({'original_first' : False})
        followup = self.evaluate(instruction, response_A, response_B, **kwargs) # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1 # both are correct and positionally invariant
        

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
    
# Testing
if __name__ == "__main__":

    import pandas as pd
    import time
    import warnings

    warnings.filterwarnings("error")

    def test():

        from guidance import models        

        # Load sample data row
        dataset = pd.read_csv("./data/human_data.csv")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset = dataset.head(1) 
        instruction = dataset["prompt"].iloc[0]
        original_text = dataset["response_a"].iloc[0]
        mutated_text = dataset["response_b"].iloc[0]
        label = ResponseQuality.TIE if dataset["selected"].iloc[0] == "tie" else ResponseQuality.A_BETTER if dataset["original"].iloc[0] else ResponseQuality.B_BETTER

        # Initialize Base LLM
        print("Initializing Base LLM with Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf")
        model_id = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-WQE-0.1-q8_0.gguf"
        llm = models.LlamaCpp(
            model=model_id,
            echo=False,
            n_gpu_layers=-1,
            n_ctx=2048
        )

        oracle = MutationOracle(llm, explain=False)

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

        # # Initialize Base LLM - FHC: THIS DOES NOT WORK RELIABLY RIGHT NOW. 
        # print("Initializing Oracle with gpt-4o-mini...")
        # model_id = "gpt-4o-mini"
        # llm = models.OpenAI(
        #     model=model_id
        # )

        # oracle = SoloOracle(llm, explain=False)

        # # Run quality assessments
        # start = time.time()
        # quality_eval = oracle.is_quality_preserved(
        #     instruction=instruction, 
        #     original_text=original_text, 
        #     mutated_text=mutated_text, 
        #     reference_answer=None
        # )
        # delta = time.time() - start
        # print("EVAL oracle.is_quality_preserved")
        # print("quality_eval:", quality_eval)
        # print("time_taken:", delta)
        
        # print("EVAL  oracle.test:")
        # start = time.time()
        # results = oracle.test(instruction, original_text, mutated_text, label)
        # delta = time.time() - start
        # print(results)
        

    test()
    