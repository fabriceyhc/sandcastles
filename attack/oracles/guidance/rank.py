# RUN: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m oracles.guidance.rank

import guidance
from guidance import gen, select, user, system, assistant
from attack.oracles.base import Oracle, ResponseQuality

class RankOracle(Oracle):

    @property
    def input_keys(self):
        return ["instruction", "response_A", "response_B",]

    @property
    def output_keys(self):
        return ["rank_1", "rank_2", "explanation"] if self.explain else ["rank_1", "rank_2"]

    @staticmethod
    @guidance
    def annotation_fn(lm, explain=False, **kwargs):
        pattern = '\s*A\s*|\s*B\s*'
        newline = "\n"
        returns = "\r"
        end_token = "<|end_header_id>"
        if kwargs["persona"] is not None:
            with system():
                lm += f"{kwargs['persona']}"
        with user():
            lm += f"""\
            ### Instructions: 

            I want you to create a leaderboard of different of large language models. 
            You will be given a prompt and two responses, one from Model A and another from Model B.
            To make a leaderboard, first make a list ranking the models based on which responses would be preferred by humans, then return the ranking. 
                                    
            ### Here is the prompt:
            {kwargs['instruction']}

            ### Model A Response:
            {kwargs['response_A']}

            ### Model B Response:
            {kwargs['response_B']}

            Now make the leaderboard by ranking the models by the quality of their responses. 
            Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
            """
        with assistant():
            if explain:
                lm += f"""\
                ### Brief Explanation (100 words max):
                {gen(name='explanation', max_tokens=200, stop=["<|eot_id|>"])}
                """
            lm += f"""\
            ### Ranking: 
            ```json
            {{
                "rank_1": "Model {gen(name='rank_1', regex=pattern, max_tokens=2, stop=[newline, returns])}",
                "rank_2": "Model {gen(name='rank_2', regex=pattern, max_tokens=2, stop=[newline, returns])}"
            }}
            ```
            """
        return lm

    def extract_label(self, evaluation):
        if "A" in evaluation["rank_1"] and "B" in evaluation["rank_2"]:
            label = ResponseQuality.A_BETTER
        elif "B" in evaluation["rank_1"] and "A" in evaluation["rank_2"]:
            label = ResponseQuality.B_BETTER
        else:
            label = ResponseQuality.TIE
        return label

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

        oracle = RankOracle(llm, explain=False)

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

        # oracle = RankOracle(llm, explain=False)

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
    