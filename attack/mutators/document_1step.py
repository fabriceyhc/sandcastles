import random
import nltk
from nltk.tokenize import sent_tokenize
import guidance
from guidance import models, gen, select, user, assistant
import hydra
import logging
from .document_2step import StringTokenLength

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def extract_dict(output, keys):
    return {k: output[k] for k in keys}

class Document1StepMutator:  
    # NOTE: This current implementation is slow (~300 seconds) and must be optimized before use in the attack. 
    # One idea would be to have it suggest the edits in some structured format and then apply them outside of generation. 
    # This prevents it from having to copy / paste over the bulk of the response unchanged. 
    def __init__(self, llm = None) -> None:
        self.llm = self._initialize_llm(llm)

        # Check if NLTK data is downloaded, if not, download it
        self._ensure_nltk_data()

    def _initialize_llm(self, llm):
        if not isinstance(llm, (models.LlamaCpp, models.OpenAI)):
            log.info("Initializing a new Mutator model...")
            llm = models.LlamaCpp(
                model="/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf",
                echo=False,
                n_gpu_layers=-1,
                n_ctx=2048*2
            )
        return llm

    def _ensure_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt') 

    def mutate(self, text):
        output = self.llm + paraphrase(text)
        #print(f"output: {output}")
        return output["paraphrase_text"].strip()


@guidance
def paraphrase(lm, text):
    with user():
        lm += f"""\
        ### Task Description: 
        Paraphrase the following text so that it preserves the meaning, quality, and format while varying the underlying words and sentence structures. 
        Keep the text at a similar length. Respond with just the new version of the text.

        ### The original text: 
        {text}
        """
    with assistant():
        lm += f"""\
        ### Paraphrased text:
        {gen('paraphrase_text', max_tokens=int(StringTokenLength.length(text) * 1.25), temperature=.6, stop=['<|im_end|>', '|im_end|>'])}
        """
    return lm


if __name__ == "__main__":

    def test():
        import time
        from utils import diff
        import pandas as pd

        print(f"Starting mutation...")

        dataset = pd.read_csv("./data/WQE/dev.csv")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        n=1
        avg_time=0
        dataset = dataset.head(n) 
        text_mutator = Document1StepMutator()
        for index, row in dataset.iterrows():
          text = row["response_a"]

          start = time.time()
          mutated_text = text_mutator.mutate(text)
          delta = time.time() - start

          print(f"Original text: {text}")
          print(f"Mutated text: {mutated_text}")
          print(f"Diff: {diff(text, mutated_text)}")
          print(f"Time taken: {delta}")
          avg_time += delta
        print(f"Average time: {avg_time/n}")

    test()