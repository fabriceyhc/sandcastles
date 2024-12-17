from transformers import pipeline
import random
import re
import string
import pandas as pd
import difflib
import torch
import logging
import os
from itertools import chain, zip_longest
from attack.utils import is_bullet_point

log = logging.getLogger(__name__)

class WordMutator:
    def __init__(self, model_name="FacebookAI/roberta-large"):
        self.model_name = model_name
        self.max_length = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fill_mask = pipeline(
            "fill-mask", 
            model=self.model_name, 
            tokenizer=self.model_name,
            device=self.device
        )
        self.tokenizer_kwargs = {"truncation": True, "max_length": 512}

    def get_words(self, text):
        split = re.split(r'(\W+)', text)
        words = split[::2]
        punctuation = split[1::2]
        return words, punctuation

    def select_random_segment(self, words, punctuation):
        if len(words) <= self.max_length:
            # Ensure lengths match
            if len(punctuation) < len(words):
                punctuation.append('')
            return words, punctuation, 0, len(words)
        
        max_start = len(words) - self.max_length
        start_index = random.randint(0, max_start)
        end_index = start_index + self.max_length

        segment_words = words[start_index:end_index]
        segment_punct = punctuation[start_index:end_index]

        # Adjust punctuation length if necessary
        if len(segment_punct) < len(segment_words):
            segment_punct.append('')

        return segment_words, segment_punct, start_index, end_index

    def intersperse_lists(self, list1, list2):
        flattened = chain.from_iterable((x or '' for x in pair) for pair in zip_longest(list1, list2))
        return ''.join(flattened)

    def mask_random_word(self, words, punctuation):
        if not words:
            return words, None, None

        found_nice_word = False
        index_to_mask = None

        attempt_count = 0
        max_attempts = 100

        while not found_nice_word and attempt_count < max_attempts:
            index_to_mask = random.randint(0, len(words) - 1)
            word_to_mask = words[index_to_mask]

            if word_to_mask and not is_bullet_point(word_to_mask):
                found_nice_word = True
            attempt_count += 1

        if not found_nice_word:
            return words, None, None

        # Mask the word
        words_with_mask = words.copy()
        words_with_mask[index_to_mask] = self.fill_mask.tokenizer.mask_token
        masked_text = self.intersperse_lists(words_with_mask, punctuation)

        return masked_text, word_to_mask, index_to_mask

    def mutate(self, text, num_replacements=0.001):
        words, punctuation = self.get_words(text)

        if len(words) > self.max_length:
            segment, seg_punc, start, end = self.select_random_segment(words, punctuation)
        else:
            segment, seg_punc, start, end = words, punctuation, 0, len(words)
            # Ensure lengths match
            if len(seg_punc) < len(segment):
                seg_punc.append('')

        if num_replacements < 0:
            raise ValueError("num_replacements must be larger than 0!")
        if 0 < num_replacements < 1:
            num_replacements = max(1, int(len(segment) * num_replacements))

        log.info(f"Making {num_replacements} replacements to the input text segment.")

        replacements_made = 0
        while replacements_made < num_replacements:
            masked_text, word_to_mask, index_to_mask = self.mask_random_word(segment, seg_punc)

            if word_to_mask is None or index_to_mask is None:
                log.warning("No valid word found to mask!")
                break

            log.info(f"Masked word: {word_to_mask}")

            # Print the masked text for debugging
            log.info(f"Masked text: {masked_text}")

            # Ensure that the mask token is present
            if self.fill_mask.tokenizer.mask_token not in masked_text:
                log.warning("Mask token not found in masked_text")
                continue

            # Use fill-mask pipeline
            candidates = self.fill_mask(masked_text, top_k=3, tokenizer_kwargs=self.tokenizer_kwargs)

            if not candidates:
                log.warning("No candidates returned from fill-mask pipeline")
                continue

            suggested_replacement = self.get_highest_score_index(candidates, blacklist=[word_to_mask.lower()])

            # Ensure valid replacement
            if suggested_replacement is None or not re.fullmatch(r'\w+', suggested_replacement['token_str'].strip()):
                log.info(f"Skipping replacement: {suggested_replacement['token_str'] if suggested_replacement else 'None'}")
                continue

            log.info(f"word_to_mask: {word_to_mask}")
            log.info(f"suggested_replacement: {suggested_replacement['token_str']} (score: {suggested_replacement['score']})")

            # Replace the masked word in the segment using the index
            segment[index_to_mask] = suggested_replacement['token_str'].strip()
            replacements_made += 1

        # Ensure punctuation lengths match for reconstruction
        punct_before = punctuation[:start]
        if len(punct_before) < len(words[:start]):
            punct_before.append('')

        punct_after = punctuation[end:]
        if len(punct_after) < len(words[end:]):
            punct_after.append('')

        if len(seg_punc) < len(segment):
            seg_punc.append('')

        # Reconstruct the text
        combined_text = self.intersperse_lists(words[:start], punct_before) + \
                        self.intersperse_lists(segment, seg_punc) + \
                        self.intersperse_lists(words[end:], punct_after)

        return self.cleanup(combined_text)


    def get_highest_score_index(self, suggested_replacements, blacklist):
        filtered_data = [d for d in suggested_replacements if d['token_str'].strip().lower() not in blacklist]

        if filtered_data:
            highest_score_index = max(range(len(filtered_data)), key=lambda i: filtered_data[i]['score'])
            return filtered_data[highest_score_index]
        else:
            return None

    def cleanup(self, text):
        return text.replace("<s>", "").replace("</s>", "")

    def diff(self, text1, text2):
        text1_lines = text1.splitlines()
        text2_lines = text2.splitlines()
        d = difflib.Differ()
        diff = list(d.compare(text1_lines, text2_lines))
        diff_result = '\n'.join(diff)
        return diff_result


def test():

    import time
    import textwrap
    import os
   
    # text = textwrap.dedent("""
    #     Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
    #     The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
    #     Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
    #     However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
    #     In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
    # """)

    text = textwrap.dedent(""""What a fascinating individual! Let's dive into the psychological portrait of someone who values their time at an astronomical rate.

**Name:** Tempus (a nod to the Latin word for ""time"")

**Profile:**

Tempus is a highly successful and ambitious individual who has cultivated a profound appreciation for the value of time. Having made their fortune through savvy investments, entrepreneurial ventures, or high-stakes decision-making, Tempus has come to regard their time as their most precious asset.

**Core traits:**

1. **Frugal with time:** Tempus guards their schedule like Fort Knox. Every moment, no matter how small, is meticulously accounted for. They prioritize tasks with ruthless efficiency, ensuring maximum productivity while minimizing idle time.
2. **Opportunity cost obsession:** When making decisions, Tempus always weighs the potential benefits against the opportunity costs – not just in financial terms but also in terms of the time required. If a task doesn't yield substantial returns or value, it's quickly discarded.
3. **Time- compression mastery:** Tempus excels at streamlining processes, leveraging technology, and optimizing routines to minimize time expenditure. Their daily routine is honed to perfection, leaving room only for high-yield activities.
4. **Profound sense of self-worth:** Tempus knows their worth and won't compromise on it. They wouldn't dream of wasting their valuable time on menial tasks or engaging in unnecessary social niceties that don't provide tangible returns.

**Behavioral patterns:**

1. **Rapid-fire decision-making:** Tempus makes swift, calculated choices after considering the time investment vs. potential ROI (return on investment). This approach often catches others off guard, as they prioritize decisive action over protracted deliberation.
2. **Minimalist scheduling:** Meetings are kept concise, and phone calls are optimized for brevity. Tempus schedules meetings back-to-back to ensure their day is filled, leaving little room for casual conversation or chit-chat.
3. **Value-driven relationships:** Personal connections are assessed by their utility and alignment with Tempus' goals. Friendships and collaborations must yield tangible benefits or offer significant learning opportunities; otherwise, they may be relegated to a peripheral role or terminated.
4. **Unflinching efficiency:** When given a task or project, Tempus approaches it with laser-like focus, aiming to deliver results within minutes (or even seconds) rather than days or weeks. Procrastination is nonexistent in their vocabulary.

**Thought patterns and values:**

1. **Economic time value theory:** Tempus intuitively calculates the present value of future time commitments, weighing pros against cons to make informed decisions.
2. **Scarcity mentality:** Time is perceived as a finite resource, fueling their quest for extraordinary productivity.
3. **Meritocratic bias:** Tempus allocates attention based solely on meritocracy – i.e., people and ideas worth investing their time in have demonstrated excellence or promise substantial ROI.
4. **High-stakes resilience:** A true entrepreneurial spirit drives Tempus, embracing calculated risks and adapting rapidly to shifting circumstances.

**Weaknesses:**

1. **Insufficient relaxation and leisure:** Overemphasizing productivity might lead Tempus to neglect essential downtime, eventually causing burnout or exhaustion.
2. **Limited patience:** Tempus can become easily frustrated when forced to engage with inefficient or incompetent individuals.
3. **Difficulty empathizing:** Prioritizing logic and outcomes may sometimes cause them to overlook or misunderstand emotional nuances.
4. **Inflexible planning:** An unyielding adherence to precision-planned timetables could result in difficulties adjusting to unexpected setbacks or spontaneous opportunities.

Overall, Tempus embodies a fierce dedication to time optimization and resource management, fueled by unwavering confidence in their self-worth and capabilities.""")

    text_mutator = WordMutator()

    start = time.time()

    df = pd.read_csv('/data2/borito1907/impossibility-watermark/data/WQE_adaptive/dev.csv')

    mutations_file_path = '/data2/borito1907/impossibility-watermark/inputs/word_mutator/test_1new.csv'

    for idx, row in df.head(50).iterrows():
        mutated_text = row['text']

        words, punct = text_mutator.get_words(mutated_text)

        log.info(f"Words: {words}")
        log.info(f"Punct: {punct}")

        for _ in range(20):
            mutated_text = text_mutator.mutate(mutated_text)
        delta = time.time() - start

        log.info(f"Original text: {text}")
        log.info(f"Mutated text: {mutated_text}")
        log.info(f"Original == Mutated: {text == mutated_text}")
        # log.info(f"Diff: {text_mutator.diff(text, mutated_text)}")
        log.info(f"Time taken: {delta}")

        stats = [{'id': row.id, 'text': row.text, 'zscore' : row.zscore, 'watermarking_scheme': row.watermarking_scheme, 'model': row.model, 'gen_time': row.time, 'mutation_time': delta, 'mutated_text': mutated_text}]
        return stats

if __name__ == "__main__":
    test()