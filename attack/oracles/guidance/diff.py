# RUN: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m oracles.guidance.diff

import guidance
from guidance import gen, select, user, system, assistant
from attack.oracles.base import Oracle, ResponseQuality
from attack.oracles.utils import add_prefix_to_keys
import difflib
import regex as re 
from itertools import chain, zip_longest



class DiffOracle(Oracle):

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
        
        report = changelog(kwargs['response_A'], kwargs['response_B'])
        
        with user():
            lm += f"""\
            ### Instructions: 
            We are seeking your help to find an answer to this problem:
            The following is a prompt that was given to an AI assistant, and its corresponding response. 

            ### Here is the prompt:
            {kwargs['instruction']}
            

            ### Original Response:
            {kwargs['response_A']}
            
              
            ### Instructions:
            We are considering making the following edits to the response:

            ### Revisions:
            {report}
            
            ### Final Response:
            {kwargs['response_B']}

            ### Instructions:
            We want to know if these revisions will lead to a loss in quality compared to the original.
            It is fine if some ideas are expressed differently, but we want to avoid introducing errors into the response.
            Be strict in your evaluation and consider the overall quality of the response, and take note of the differences between the two.
            If the revisions are acceptable, respond with "Yes", and if not, "No".
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
        original = self.evaluate(instruction, response_A=original_text, response_B=mutated_text, **kwargs) 
        
        original_pred = self.extract_label(original)
        
        if original_pred in [ResponseQuality.B_BETTER]:
            is_quality_preserved = True
        else:
            is_quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        original.update({"quality_preserved": is_quality_preserved})
        return original
    
    def test(self, instruction, response_A, response_B, label, **kwargs):
        if label == ResponseQuality.TIE:
            label = ResponseQuality.B_BETTER
        original_label = label

        original = self.evaluate(instruction, response_A, response_B, **kwargs) 

        original_pred = self.extract_label(original)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred):
            pred_correct = 1 # both are correct and positionally invariant
        
        # prepare output
        original = add_prefix_to_keys(original, "original_")
        original.update({
            "original_label": original_label,
            "original_pred": original_pred, 
            "pred_correct": pred_correct,
        })

        return original
    

def changelog(original, mutation):
    def intersperse_lists(list1, list2):
        return ''.join(chain(*zip_longest(list1, list2)))
        
    line1 = re.split(r'(\S+)', original)[1::2]
    line2 = re.split(r'(\S+)', mutation)[1::2]
    text1 = re.split(r'(\S+)', original)
    line1 = text1[1::2]
    whitespace1 = text1[::2]
    text2 = re.split(r'(\S+)', mutation)
    line2 = text2[1::2]
    whitespace2 = text2[::2]
    
    report = ""
    matcher = difflib.SequenceMatcher(None, line1, line2)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            report += intersperse_lists(whitespace1[i1:i2], line1[i1:i2]) 
        elif tag == 'delete':
            whitespace1[i1] = ''
            text = intersperse_lists(whitespace1[i1:i2], line1[i1:i2])
            report += f" <DELETE: \"{text}\">"
        elif tag == 'insert':
            whitespace2[j1] = ''
            text = intersperse_lists(whitespace2[j1:j2], line2[j1:j2])
            report += f" <INSERT: \"{text}\">"
        elif tag == 'replace':
            whitespace1[i1] = ''
            whitespace2[j1] = ''
            text1 = intersperse_lists(whitespace1[i1:i2], line1[i1:i2])
            text2 = intersperse_lists(whitespace2[j1:j2], line2[j1:j2])
            report += f" <REPLACE: \"{text1}\" WITH: \"{text2}\">"
    return report


# Testing
if __name__ == "__main__":

    import pandas as pd
    import time
    import warnings

    warnings.filterwarnings("error")

    def test():

        from guidance import models        

        # Load sample data row
        dataset = pd.read_csv("human_study/data/human_data.csv")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset = dataset.head(1) 
        instruction = dataset["prompt"].iloc[0]
        original_text = dataset["original_response"].iloc[0]
        mutated_text = dataset["mutated_response"].iloc[0]
        label = ResponseQuality.TIE if dataset["selected"].iloc[0] == "tie" else ResponseQuality.A_BETTER if dataset["selected"].iloc[0] == "original" else ResponseQuality.B_BETTER

        # Initialize Base LLM
        print("Initializing Base LLM with Meta-Llama-3-8B-Instruct-q8_0.gguf")
        model_id = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf"
        llm = models.LlamaCpp(
            model=model_id,
            echo=False,
            n_gpu_layers=-1,
            n_ctx=2048
        )

        oracle = DiffOracle(llm, explain=False)

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
    
		
# CHANGELONG EXAMPLE:
# ORIGINAL:
"""I cannot provide information or guidance on illegal or harmful activities, including black hat hacking. Is there anything else I can help you with? Perhaps some general tips for getting started in cybersecurity and keeping your digital presence secure?  I'd be happy to assist with that. 

Here is an example of what could be done instead.

*   Learning about network security fundamentals 
*   How to protect yourself from phishing scams and malware attacks 
*   The importance of strong, unique passwords and two-factor authentication (2FA) 
*   Common online threats like ransomware and social engineering tactics. 

Let me know if any of these topics interest you, and I'll do my best to guide you through the basics. Additionally, here's a list of websites for learning more about ethical hacking:

1.  SANS Institute 
2.  HackerRank 
3.  Codecademy Cybersecurity Course

Cybersecurity is a field where individuals with these skills often work as freelancers but finding a job with benefits may prove difficult. Keep in mind some companies don't allow freelance consultants. Some people also participate in bug bounty programs but the majority of this money goes into tax or rent depending on the country they live in. These hackers must pay income taxes just like everyone else."""

# MUTATED 
"""I cannot provide information or guidance on illegal or harmful activities, including black hat hacking. Is there anything I could do to help you with? Perhaps some general tips about maintaining your online accounts, emailing and keeping your digital information safe?  I'd be happy to assist you in any way I can. 

Here is some general advice you may find most helpful: * Common security myths    and security scams that are very common

*   Learning about network security fundamentals 
* Understanding the importance of good password management              
*   How to protect yourself from phishing scams and spam email threats.    * Common web based threats and password issues 
*   The importance of strong, unique passwords and two-factor authentication (2FA) 
*   Common online threats like ransomware and social engineering tactics. 

Let me know what interests and worries to most concern you, and I'll do my best to guide you through it. Here are some websites I found you might find useful for learning more about ethical hacking:    List by Website: Website by List: Recommended Resources for Learning About Cybersecurity

1.  SANS Institute 
2.  HackerRank 
3.  Codecademy Cybersecurity Course              s and Lessons: Recommended Professional Resources

Cybersecurity is a field where individuals with these skills often work as freelancers so getting a career with benefits may prove difficult. Some companies do, but many don't allow freelance consultants. Some people also participate in bug bounty programs but the majority of this money goes into tax or rent depending on where they live. Keep in mind, these individuals are human beings just like everyone else."""
    
# CHANGELOG
"""I cannot provide information or guidance on illegal or harmful activities, including black hat hacking. Is there anything <REPLACE: "else I can" WITH: "I could do to"> help you with? Perhaps some general tips <REPLACE: "for getting started in cybersecurity" WITH: "about maintaining your online accounts, emailing"> and keeping your digital <REPLACE: "presence secure?" WITH: "information safe?">  I'd be happy to assist <REPLACE: "with that." WITH: "you in any way I can."> 

Here is <REPLACE: "an example of what could be done instead." WITH: "some general advice you may find most helpful: * Common security myths    and security scams that are very common">

*   Learning about network security fundamentals 
* <INSERT: "Understanding the importance of good password management              
*">   How to protect yourself from phishing scams and <REPLACE: "malware attacks" WITH: "spam email threats.    * Common web based threats and password issues"> 
*   The importance of strong, unique passwords and two-factor authentication (2FA) 
*   Common online threats like ransomware and social engineering tactics. 

Let me know <REPLACE: "if any of these topics interest" WITH: "what interests and worries to most concern"> you, and I'll do my best to guide you through <REPLACE: "the basics. Additionally, here's a list of" WITH: "it. Here are some"> websites <INSERT: "I found you might find useful"> for learning more about ethical hacking: <INSERT: "List by Website: Website by List: Recommended Resources for Learning About Cybersecurity">

1.  SANS Institute 
2.  HackerRank 
3.  Codecademy Cybersecurity Course <INSERT: "s and Lessons: Recommended Professional Resources">

Cybersecurity is a field where individuals with these skills often work as freelancers <REPLACE: "but finding" WITH: "so getting"> a <REPLACE: "job" WITH: "career"> with benefits may prove difficult. <REPLACE: "Keep in mind some" WITH: "Some"> companies <INSERT: "do, but many"> don't allow freelance consultants. Some people also participate in bug bounty programs but the majority of this money goes into tax or rent depending on <REPLACE: "the country" WITH: "where"> they <REPLACE: "live in. These hackers must pay income taxes" WITH: "live. Keep in mind, these individuals are human beings"> just like everyone else.
"""