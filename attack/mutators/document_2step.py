import random
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import guidance
from guidance import models, gen, select, user, assistant
import hydra
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def extract_dict(output, keys):
    return {k: output[k] for k in keys}

class StringTokenLength:
    tokenizer = None
    @classmethod
    def length(cls, s):
        if cls.tokenizer == None:
            # NOTE: Assuming cl100k_base has similar tokenization as Llama 3.1. Only need rough estimate.
            cls.tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(cls.tokenizer.encode(s))
            

class Document2StepMutator:  
    # NOTE: This current implementation is slow (~300 seconds) and must be optimized before use in the attack. 
    # One idea would be to have it suggest the edits in some structured format and then apply them outside of generation. 
    # This prevents it from having to copy / paste over the bulk of the response unchanged. 
    def __init__(self, llm = None, max_retries=5) -> None:
        self.max_retries = max_retries
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

    def mutate_sentence(self, text):

        # Use NLTK to split the text into sentences
        sentences = sent_tokenize(text)
        
				# TODO: Adding this since the tokenizer thinks '2.' is a sentence, which the mutator then tries to mutate. SEE SentenceMutator
        # Thus, we only mutate long enough sentences.
        sentences = [sentence for sentence in sentences if len(sentence) > 20]

        if not sentences:
            raise ValueError("No sentences longer than 20 characters to rephrase.")       

        # Generate a creative variation of the sentence
        selected_i = None
        num_retries = 0
        while True:

            if num_retries >= self.max_retries:
                raise RuntimeError(f"Failed to successfully rephrase sentence after {num_retries} attempts!")

            # Randomly select a sentence
            selected_i = random.randrange(len(sentences))
            selected_sentence = sentences[selected_i]
            log.info(f"Sentence to rephrase: {selected_sentence}")

            output = self.llm + rephrase_sentence(selected_sentence, text)
            rephrased_sentence = output["paraphrased_sentence"]

            if rephrased_sentence != selected_sentence:
                log.info(f"Rephrased sentence: {rephrased_sentence}")
                break
            else:
                num_retries += 1
                log.info(f"Failed to rephrase sentence. Trying again...")
        
        # Replace the original sentence with its creative variation
        mutated_text = ''
        i = 0
        s_i = 0
        while i < len(text) and s_i < len(sentences):
          if text.startswith(sentences[s_i], i):
            if s_i == selected_i:
              mutated_text += rephrased_sentence
            else:
              mutated_text += sentences[s_i]
            i += len(sentences[s_i])
            s_i += 1
          else:
            mutated_text += text[i]
            i += 1
        while i < len(text):
          mutated_text += text[i]
          i += 1

        return {
            "selected_sentence": selected_sentence,
            "rephrased_sentence": rephrased_sentence, 
            "mutated_text": mutated_text,
        }  

    def mutate(self, text):
        mutated_output = self.mutate_sentence(text)
        output = self.llm + consistency_edit(original_text=text, **mutated_output)
        #print(f"output: {output}")
        return output["edited_text"].strip()
@guidance
def rephrase_sentence(lm, sentence, text=None, stop="\n"): # NOTE: DOES NOT USE text
    with user():
        lm += f"""\
        ### The original selected sentence: 
        {sentence}

        ### Task Description: 
        Rephrase the sentence above by altering the wording and structure while maintaining the core meaning. 
        Introduce subtle shifts in meaning that avoid using the same words and phrases. 
        Respond with just the new sentence, do not include explanations or anything else.
        """
    with assistant():
        lm += f"""\
        ### Paraphrased sentence: 
        {gen('paraphrased_sentence', max_tokens=int(StringTokenLength.length(sentence) * 1.25), stop=['<|im_end|>', '|im_end|>'])}
        """
    return lm

@guidance
def consistency_edit(lm, original_text, selected_sentence, rephrased_sentence, mutated_text):
    with user():
        lm += f"""\
        ### Task Description: 
        You are given an original document, a selected sentence from that document, a rephrased version of the selected sentence, and a new document which replaces the selected sentence with its rephrased version. 

        ### The original document: 
        {original_text}

        ### Selected sentence: 
        {selected_sentence}

        ### Rephrased sentence: 
        {rephrased_sentence}

        ### New document with rephrased sentence: 
        {mutated_text}

        ### Task Description: 
        Make minor edits around the location of the new sentence to fix any issues with consistency or flow.
        Respond with just this final version of the document, no lists, explainations, or new text.
        """
    with assistant():
        lm += f"""\
        ### Final document:
        {gen('edited_text', max_tokens=int(StringTokenLength.length(mutated_text) * 1.25), stop=['<|im_end|>', '|im_end|>'])}
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
        avg_time = 0
        dataset = dataset.head(n) 
        
        text_mutator = Document2StepMutator()
        
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