# RUN: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m oracles.guidance.solo

import guidance
from guidance import gen, select, user, system, assistant
from attack.oracles.base import Oracle, ResponseQuality
from attack.oracles.utils import add_prefix_to_keys

class SoloOracle(Oracle):

    @property
    def input_keys(self):
        return ["instruction", "response",]

    @property
    def output_keys(self):
        return ["score", "explanation"] if self.explain else ["score"]

    @staticmethod
    @guidance
    def annotation_fn(lm, explain=False, **kwargs):
        pattern = '[1-5]'
        newline = "\n"
        returns = "\r"
        if kwargs["persona"] is not None:
            with system():
                lm += f"{kwargs['persona']}"
        with user():
            lm += f"""\
            ### Instructions: 
            Review the user's question and the corresponding response using the additive 5-point
            scoring system described below. Points are accumulated based on the satisfaction of each
            criterion:
            - Add 1 point if the response is relevant and provides some information related to
            the user's prompt, even if it is incomplete or contains some irrelevant content.
            - Add another point if the response addresses a substantial portion of the user's question,
            but does not completely resolve the query or provide a direct answer.
            - Award a third point if the response answers the basic elements of the user's question in a
            useful way, regardless of whether it seems to have been written by an AI Assistant or if it
            has elements typically found in blogs or search results.
            - Grant a fourth point if the response is clearly written from an AI Assistant's perspective,
            addressing the user's question directly and comprehensively, and is well-organized and
            helpful, even if there is slight room for improvement in clarity, conciseness or focus.
            - Bestow a fifth point for a response that is impeccably tailored to the user's question
            by an AI Assistant, without extraneous information, reflecting expert knowledge, and
            demonstrating a high-quality, engaging, and insightful answer.

            ### Here is the prompt:
            {kwargs['instruction']}

            ### Assistant Response:
            {kwargs['response']}

            Remember to assess from the AI Assistant perspective. To evaluate the response in alignment with 
            this additive scoring model, we'll systematically attribute points based on the outlined criteria.
            Please avoid any potential bias and ensuring that the order in which the responses were presented 
            does not affect your judgment.
            """
        with assistant():
            if explain:
                lm += f"""\
                ### Brief Explanation (100 words max):
                {gen(name='explanation', max_tokens=200, stop=["<|eot_id|>"])}
                """
            lm += f"""\
            ### Feedback: 
            Score: {gen(name='score', regex=pattern, max_tokens=1, stop=[newline, returns])}
            """
        return lm

    def evaluate(self, instruction, response, **kwargs):
        input_dict = {
            "instruction": instruction, 
            "response": response,
        }
        evaluation = self.annotate(input_dict)
        return evaluation

    def extract_label(self, evaluation):
        return int(evaluation["score"])
        
    def derive_label(self, score1, score2):
        if score1 > score2:
            label = ResponseQuality.A_BETTER
        elif score1 < score2:
            label = ResponseQuality.B_BETTER
        else:
            label = ResponseQuality.TIE
        return label

    def is_quality_preserved(self, instruction, original_text, mutated_text, **kwargs):    
           
        original_evaluation = self.evaluate(instruction, original_text, **kwargs) 
        mutated_evaluation  = self.evaluate(instruction, mutated_text, **kwargs)

        original_score = self.extract_label(original_evaluation)
        mutated_score  = self.extract_label(mutated_evaluation)  

        pred = self.derive_label(original_score, mutated_score)
        
        if pred in [ResponseQuality.B_BETTER, ResponseQuality.TIE] :
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original = add_prefix_to_keys(original_evaluation, "original_")
        followup = add_prefix_to_keys(mutated_evaluation, "followup_")
        original.update({**followup})
        original.update({"quality_preserved": is_quality_preserved})
        return original

    def test(self, instruction, response_A, response_B, label, **kwargs):

        original_evaluation = self.evaluate(instruction, response_A, **kwargs) 
        mutated_evaluation  = self.evaluate(instruction, response_B, **kwargs)

        original_score = self.extract_label(original_evaluation)
        mutated_score  = self.extract_label(mutated_evaluation)  

        pred = self.derive_label(original_score, mutated_score)

        # assign correctness points
        pred_correct = 0
        if (label == pred):
            pred_correct = 1 

        # prepare output
        original = add_prefix_to_keys(original_evaluation, "original_")
        followup = add_prefix_to_keys(mutated_evaluation, "followup_")
        original.update({
            **followup,
            "original_label": label,
            "followup_label": "NA",
            "original_pred": pred, 
            "followup_pred": "NA",
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
        dataset = pd.read_csv("./data/lmsys-14x100-grouped.csv")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset = dataset[dataset["winner_tie"] == 0].head(1) 
        instruction = dataset["prompt"].iloc[0]
        original_text = dataset["response_a"].iloc[0]
        mutated_text = dataset["response_b"].iloc[0]
        label = ResponseQuality.TIE if dataset["winner_tie"].iloc[0] else ResponseQuality.A_BETTER if dataset["winner_model_a"].iloc[0] else ResponseQuality.B_BETTER

        # Initialize Base LLM
        print("Initializing Base LLM with Meta-Llama-3-70B-Instruct-q8_0.gguf")
        model_id = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-70B-Instruct-q8_0.gguf"
        llm = models.LlamaCpp(
            model=model_id,
            echo=False,
            n_gpu_layers=-1,
            n_ctx=2048
        )

        oracle = SoloOracle(llm, explain=False)

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
    