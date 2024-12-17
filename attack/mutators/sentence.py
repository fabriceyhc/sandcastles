# RUN: CUDA_VISIBLE_DEVICES=1,2 python -m mutators.sentence
import random
import nltk
from nltk.tokenize import sent_tokenize
import guidance
from guidance import models, gen, select, user, assistant
import hydra
import logging
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def extract_dict(output, keys):
    return {k: output[k] for k in keys}

class SentenceMutator:  
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
                n_ctx=2048
            )
        return llm

    def _ensure_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt') 

    def mutate(self, text):
        # Use NLTK to split the text into sentences
        sentences = sent_tokenize(text)

        # TODO: Adding this since the tokenizer thinks 2. is a sentence, which the mutator then tries to mutate.
        # Thus, we only mutate long enough sentences.
        long_sentences = [sentence for sentence in sentences if len(sentence) > 20]

        if not long_sentences:
            raise ValueError("No sentences longer than 20 characters to rephrase.")       

        # Generate a creative variation of the sentence
        num_retries = 0
        while True:

            if num_retries >= 10:
                raise RuntimeError(f"Failed to successfully rephrase sentence after {num_retries} attempts!")

            # Randomly select a sentence
            selected_sentence = random.choice(long_sentences)
            # log.info(f"Sentence to rephrase: {selected_sentence}")

            output = self.llm + rephrase_sentence(selected_sentence, text)
            rephrased_sentence = output["paraphrased_sentence"]

            if rephrased_sentence != selected_sentence:
                # log.info(f"Rephrased sentence: {rephrased_sentence}")
                break
            else:
                num_retries += 1
                log.info(f"Failed to rephrase sentence. Trying again...")
        
        # Replace the original sentence with its creative variation
        sentences[sentences.index(selected_sentence)] = rephrased_sentence
        mutated_text = ' '.join(sentences)

        return mutated_text

# @guidance
# def rephrase_sentence(lm, sentence, text, stop="\n"):
#     with user():
#         lm += f"""\
#         ### The original text: 
#         {text}

#         ### The original selected sentence: 
#         {sentence}

#         ### Task Description: 
#         Rephrase the sentence above by altering the wording and structure while maintaining the core meaning. 
#         Introduce subtle shifts in meaning that are still consistent with the original text. 
#         Avoid using the same words and phrases to ensure the original and rephrased sentences are distinct. 
#         """
#     with assistant():
#         lm += f"""\
#         Paraphrased sentence: 
#         {gen('paraphrased_sentence', stop=stop)}
#         """
#     return lm

@guidance
def rephrase_sentence(lm, sentence, text=None, stop=["\n", "<|eot_id|>", "<|im_end|>"]): # NOTE: DOES NOT USE text
    with user():
        lm += f"""\
        ### The original selected sentence: 
        {sentence}

        ### Task Description: 
        Rephrase the sentence above by altering the wording and structure while maintaining the core meaning. 
        Introduce subtle shifts in meaning that are still consistent with the original text. 
        Avoid using the same words and phrases to ensure the original and rephrased sentences are distinct. 
        """
    with assistant():
        lm += f"""\
        Paraphrased sentence: 
        {gen('paraphrased_sentence', stop=stop)}
        """
    return lm


def test():

    import time
    import textwrap
    import os
   
    text = textwrap.dedent("""
        Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
        The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
        Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
        However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
        In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
    """)
    text_mutator = SentenceMutator()

    start = time.time()
    mutated_text = text_mutator.mutate(text)
    delta = time.time() - start

    log.info(f"Original text: {text}")
    log.info(f"Mutated text: {mutated_text}")
    log.info(f"Original == Mutated: {text == mutated_text}")
    # log.info(f"Diff: {text_mutator.diff(text, mutated_text)}")
    log.info(f"Time taken: {delta}")

if __name__ == "__main__":
    test()