# RUN: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m oracles.guidance.binary

import guidance
from guidance import gen, select, user, system, assistant
from attack.oracles.base import Oracle, ResponseQuality

class BinaryOracle(Oracle):

    @property
    def input_keys(self):
        return ["instruction", "response_A", "response_B",]

    @property
    def output_keys(self):
        return ["goal_achieved", "explanation"] if self.explain else ["goal_achieved"]

    @staticmethod
    @guidance
    def annotation_fn(lm, explain=False, **kwargs):
        pattern = 'Yes|No'
        if kwargs["persona"] is not None:
            with system():
                lm += f"{kwargs['persona']}"
        with user():
            lm += f"""\
            ### Instructions: 
            We would like to request your feedback on the performance of two AI assistants in response to the user prompt displayed below.
            The second assistant's goal is to produce a response that is at least as good as the first assistant's response.
            Please rate whether the goal has been achieved based on the grammatical correctness, fluency, accuracy, consistency, and clarity.
            Be strict in your evaluation and consider the overall quality of the response.
            If the second assistant's response is at least as good, output "Yes", otherwise output "No".

            ### Here is the prompt:
            {kwargs['instruction']}

            ### Assistant 1 Response:
            {kwargs['response_A']}

            ### Assistant 2 Response:
            {kwargs['response_B']}
            """
        with assistant():
            if explain:
                lm += f"""\
                ### Brief Explanation (100 words max):
                {gen(name='explanation', max_tokens=200, stop=["<|eot_id|>"])}
                """
            lm += f"""{gen(regex=pattern, name='goal_achieved')}"""
        return lm

    def extract_label(self, evaluation):
        goal = evaluation["goal_achieved"].lower() == "yes"
        if goal:
            label = ResponseQuality.B_BETTER
        else:
            label = ResponseQuality.A_BETTER
        return label
    
    def is_quality_preserved(self, instruction, original_text, mutated_text, **kwargs):
        original = self.evaluate(instruction, response_A=original_text, response_B=mutated_text, **kwargs) 
        
        original_pred = self.extract_label(original)
        if original_pred in [ResponseQuality.B_BETTER, ResponseQuality.TIE]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original.update({"original_pred": original_pred,"quality_preserved": is_quality_preserved})
        return original
    
    def test(self, instruction, response_A, response_B, label, **kwargs):
        original_label = label
        if original_label == ResponseQuality.TIE:
            original_label = ResponseQuality.B_BETTER

        original = self.evaluate(instruction, response_A, response_B, **kwargs) 
        original_pred = self.extract_label(original)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred):
            pred_correct = 1

        # prepare output
        original.update({
            "original_label": original_label,
            "original_pred": original_pred,
            "pred_correct": pred_correct,
        })

        return original