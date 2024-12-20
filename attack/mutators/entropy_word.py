# CUDA_VISIBLE_DEVICES=5,6 python -m mutators.entropy_word

import random
import re
import pandas as pd
import difflib
import torch
import logging
import os
from itertools import chain, zip_longest
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from attack.utils import is_bullet_point

log = logging.getLogger(__name__)

def compute_token_entropies_and_word_ids(segment_words, model, tokenizer):
    model.eval()
    with torch.no_grad():
        # Tokenize the words with is_split_into_words=True
        encoding = tokenizer(
            segment_words,
            is_split_into_words=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        input_ids = encoding['input_ids'].to(model.device)
        word_ids = encoding.word_ids(batch_index=0)
        
        # Get model outputs
        outputs = model(input_ids)
        logits = outputs.logits.cpu()
                
        # Compute probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
        # Compute token entropies
        token_entropies = []
        for idx in range(len(input_ids[0])):
            prob_dist = probs[0, idx, :]
            token_entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))
            token_entropies.append(token_entropy.item())

    return token_entropies, word_ids

def aggregate_entropies_per_word(token_entropies, word_ids):
    entropies = []  # Stores the average entropy for each word.
    current_word_id = None  # Tracks the current word ID during iteration.
    current_word_entropy = []  # Stores the entropies for tokens belonging to the same word.

    for idx, word_id in enumerate(word_ids):
        if word_id is None:  # Skip tokens that are not part of any word (e.g., padding or special tokens).
            continue

        # Check if we have moved to a new word
        if word_id != current_word_id:
            if current_word_entropy:
                # Calculate the average entropy for the previous word
                entropies.append(sum(current_word_entropy) / len(current_word_entropy))
                current_word_entropy = []  # Reset the list for the new word.

            # Update the current word ID to the new word.
            current_word_id = word_id

        # Add the entropy of the current token to the list.
        current_word_entropy.append(token_entropies[idx])

    # After the loop, handle the final word (if any tokens were collected for it).
    if current_word_entropy:
        entropies.append(sum(current_word_entropy) / len(current_word_entropy))

    return entropies

def compute_entropies(text, model, tokenizer):
    token_entropies, word_ids = compute_token_entropies_and_word_ids(text, model, tokenizer)
    entropies = aggregate_entropies_per_word(token_entropies, word_ids)
    
    return entropies

class EntropyWordMutator:
    def __init__(self, model_name="FacebookAI/roberta-large", measure_model_name="EleutherAI/gpt-neo-2.7B"):
        self.model_name = model_name
        self.max_length = 256
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.fill_mask = pipeline(
            "fill-mask", 
            model=self.model_name, 
            tokenizer=self.model_name,
            device=self.device
        )
        self.tokenizer_kwargs = {"truncation": True, "max_length": 512}

        self.measure_model_name = measure_model_name

        self.measure_tokenizer = AutoTokenizer.from_pretrained(self.measure_model_name, add_prefix_space=True)
        self.measure_model = AutoModelForCausalLM.from_pretrained(self.measure_model_name, device_map='auto', cache_dir="/data2/.shared_models")
        self.measure_model.eval()

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
    
    def mask_word_using_candidate_indices(self, words, punctuation, candidate_indices):
        """
        Attempts to mask a word from the given list of candidate indices.
        If none of the candidates work, returns None.
        """
        if not words or not candidate_indices:
            return None, None, None

        random.shuffle(candidate_indices)

        found_nice_word = False
        index_to_mask = None

        attempt_count = 0
        max_attempts = len(candidate_indices)  # Limit attempts to provided candidates

        while not found_nice_word and attempt_count < max_attempts:
            # Use the provided candidate indices to try masking
            index_to_mask = candidate_indices[attempt_count]
            word_to_mask = words[index_to_mask]

            # Check if the selected word is valid (not a bullet point)
            if word_to_mask and not is_bullet_point(word_to_mask):
                found_nice_word = True
            attempt_count += 1

        # If no suitable word was found, return None
        if not found_nice_word:
            return None, None, None

        # Mask the selected word
        words_with_mask = words.copy()
        words_with_mask[index_to_mask] = self.fill_mask.tokenizer.mask_token

        # Reconstruct the masked text with interspersed punctuation
        masked_text = self.intersperse_lists(words_with_mask, punctuation)

        return masked_text, word_to_mask, index_to_mask

    def mutate(self, text, num_replacements=0.001):
        words, punctuation = self.get_words(text)

        if len(words) > self.max_length:
            segment_words, seg_punc, start, end = self.select_random_segment(words, punctuation)
        else:
            segment_words, seg_punc, start, end = words, punctuation, 0, len(words)
            # Ensure lengths match
            if len(seg_punc) < len(segment_words):
                seg_punc.append('')


        # Compute entropies
        entropies = compute_entropies(segment_words, self.measure_model, self.measure_tokenizer)

        if len(entropies) != len(segment_words):
            log.warning(f"Entropy length and segment word length not the same!!!")

        log.info(f"Segment Words: {segment_words}")
        log.info(f"Length of the Segment Words: {len(segment_words)}")
        log.info(f"Entropies: {entropies}")
        log.info(f"Length of the Entropies: {len(entropies)}")

        # Create a list of (index, word, entropy)
        word_entropies = list(zip(range(len(segment_words)), segment_words, entropies))

        # Sort by entropy in descending order
        word_entropies_sorted = sorted(word_entropies, key=lambda x: x[2], reverse=True)

        # Get the indices of the top 20 words
        top_n = 20
        top_words = word_entropies_sorted[:top_n]
        candidate_indices = [idx for idx, _, _ in top_words]
        
        if num_replacements < 0:
            raise ValueError("num_replacements must be larger than 0!")
        if 0 < num_replacements < 1:
            num_replacements = max(1, int(len(segment_words) * num_replacements))

        log.info(f"Making {num_replacements} replacements to the input text segment.")

        replacements_made = 0
        while replacements_made < num_replacements:
            masked_text, word_to_mask, index_to_mask = self.mask_word_using_candidate_indices(segment_words, seg_punc, candidate_indices)

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
            segment_words[index_to_mask] = suggested_replacement['token_str'].strip()
            replacements_made += 1

        # Ensure punctuation lengths match for reconstruction
        punct_before = punctuation[:start]
        if len(punct_before) < len(words[:start]):
            punct_before.append('')

        punct_after = punctuation[end:]
        if len(punct_after) < len(words[end:]):
            punct_after.append('')

        if len(seg_punc) < len(segment_words):
            seg_punc.append('')

        # Reconstruct the text
        combined_text = self.intersperse_lists(words[:start], punct_before) + \
                        self.intersperse_lists(segment_words, seg_punc) + \
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

#     text = textwrap.dedent(""""What a fascinating individual! Let's dive into the psychological portrait of someone who values their time at an astronomical rate.

# **Name:** Tempus (a nod to the Latin word for ""time"")

# **Profile:**

# Tempus is a highly successful and ambitious individual who has cultivated a profound appreciation for the value of time. Having made their fortune through savvy investments, entrepreneurial ventures, or high-stakes decision-making, Tempus has come to regard their time as their most precious asset.

# **Core traits:**

# 1. **Frugal with time:** Tempus guards their schedule like Fort Knox. Every moment, no matter how small, is meticulously accounted for. They prioritize tasks with ruthless efficiency, ensuring maximum productivity while minimizing idle time.
# 2. **Opportunity cost obsession:** When making decisions, Tempus always weighs the potential benefits against the opportunity costs – not just in financial terms but also in terms of the time required. If a task doesn't yield substantial returns or value, it's quickly discarded.
# 3. **Time- compression mastery:** Tempus excels at streamlining processes, leveraging technology, and optimizing routines to minimize time expenditure. Their daily routine is honed to perfection, leaving room only for high-yield activities.
# 4. **Profound sense of self-worth:** Tempus knows their worth and won't compromise on it. They wouldn't dream of wasting their valuable time on menial tasks or engaging in unnecessary social niceties that don't provide tangible returns.

# **Behavioral patterns:**

# 1. **Rapid-fire decision-making:** Tempus makes swift, calculated choices after considering the time investment vs. potential ROI (return on investment). This approach often catches others off guard, as they prioritize decisive action over protracted deliberation.
# 2. **Minimalist scheduling:** Meetings are kept concise, and phone calls are optimized for brevity. Tempus schedules meetings back-to-back to ensure their day is filled, leaving little room for casual conversation or chit-chat.
# 3. **Value-driven relationships:** Personal connections are assessed by their utility and alignment with Tempus' goals. Friendships and collaborations must yield tangible benefits or offer significant learning opportunities; otherwise, they may be relegated to a peripheral role or terminated.
# 4. **Unflinching efficiency:** When given a task or project, Tempus approaches it with laser-like focus, aiming to deliver results within minutes (or even seconds) rather than days or weeks. Procrastination is nonexistent in their vocabulary.

# **Thought patterns and values:**

# 1. **Economic time value theory:** Tempus intuitively calculates the present value of future time commitments, weighing pros against cons to make informed decisions.
# 2. **Scarcity mentality:** Time is perceived as a finite resource, fueling their quest for extraordinary productivity.
# 3. **Meritocratic bias:** Tempus allocates attention based solely on meritocracy – i.e., people and ideas worth investing their time in have demonstrated excellence or promise substantial ROI.
# 4. **High-stakes resilience:** A true entrepreneurial spirit drives Tempus, embracing calculated risks and adapting rapidly to shifting circumstances.

# **Weaknesses:**

# 1. **Insufficient relaxation and leisure:** Overemphasizing productivity might lead Tempus to neglect essential downtime, eventually causing burnout or exhaustion.
# 2. **Limited patience:** Tempus can become easily frustrated when forced to engage with inefficient or incompetent individuals.
# 3. **Difficulty empathizing:** Prioritizing logic and outcomes may sometimes cause them to overlook or misunderstand emotional nuances.
# 4. **Inflexible planning:** An unyielding adherence to precision-planned timetables could result in difficulties adjusting to unexpected setbacks or spontaneous opportunities.

# Overall, Tempus embodies a fierce dedication to time optimization and resource management, fueled by unwavering confidence in their self-worth and capabilities.""")

    text = textwrap.dedent("""What an intriguing task!

Meet the Time Valuer, someone who has internalized the notion that their time is infinitely precious and irreplaceable. This individual's psyche is a fascinating case study in prioritization, optimization, and boundaries.

**Personality Profile:**

* **Efficiency-obsessed:** The Time Valuer is always seeking ways to streamline tasks, eliminate distractions, and maximize productivity. They are masters of minimizing time-wasting activities, as every second counts.
* **Opportunity-cost-focused:** When faced with decisions, they meticulously weigh the pros and cons, calculating the time investment required for each option. If the returns don't justify the time spent, they'll politely decline or delegate when possible.
* **Prioritization expert:** With a clear sense of purpose, this person can categorize tasks into " Must," "Should," and "Nice-to-have" lists. They execute high-priority tasks first, ensuring maximum impact on their time investment.
* **Boundary-setter extraordinaire:** Time Valuers are unapologetic about setting and maintaining healthy limits with others. They value their time too much to indulge in non-essential commitments or engage in people-pleasing behavior.
* **Time-conscious communication style:** Expect brief, informative exchanges from the Time Valuer. They'll convey essential information concisely, using every tool at their disposal (e.g., scheduling software, automated responses) to save time in interactions.

**Behavioral Patterns:**

* **Relentless calendar management:** Their digital calendars are always up-to-date, scheduled down to the minute. Overcommitting? No way! Free time is carefully allocated for relaxation, learning, or strategic planning.
* **Adopting technologies to save time:** Early adopters of innovative tools and gadgets, Time Valuers continuously seek solutions to optimize daily routines. Automating repetitive tasks, leveraging AI, and harnessing smart home devices are all part of their arsenal.
* **Avid learners (but not for leisure):** Investing time in self-improvement is vital, but only if it directly enhances their skillset or improves productivity. Expect them to consume books, podcasts, and online courses focused on business, tech, or personal development.
* **Expedient decision-making:** Without overthinking, Time Valuers make calculated choices quickly. They trust their instincts and act swiftly to capitalize on opportunities, as dithering wastes valuable seconds.

**Thought Processes:**

* **Mental math calculations:** Whenever considering a new commitment or activity, the Time Valuer intuitively calculates its potential return on investment (ROI). "Is this 5-minute conversation worth 10 million dollars?" might be an internal query.
* **Opportunity cost analysis:** "If I spend 30 minutes watching TV, what else could I have accomplished in that timeframe?" Self-reflection and weighing alternative uses for their time keeps their attention focused on goals.
* **Personal time audit:** Regular self-assessments help identify areas where time leaks occur, enabling adjustments to prevent time wastage and ensure optimal allocation.

**Emotional Landscape:**

* **Frustration tolerance:** Patience wears thin when dealing with inefficiencies, delays, or unpreparedness from others. Expect brief expressions of annoyance before redirecting focus back to more productive pursuits.
* **Contentment from accomplishment:** Achieving milestones and completing projects within allotted timeframes brings immense satisfaction, reinforcing the value placed on time.
* **No room for regret:** Prioritizing time means letting go of woulda-coulda-shouldas. Embracing choices made in the name of time optimization eliminates lingering doubts.

**Psychological Strengths:**

1. **Unyielding motivation**: Valuing time so intensely propels this individual toward success, driving their quest for continuous improvement.
2. **Effortless prioritization**: A keen sense of what truly matters enables them to allocate resources wisely, leading to remarkable accomplishments.
3. **Mastery of resilience**: Life's unexpected setbacks won't deter the Time Valuer; they swiftly adapt, recalculating plans to preserve forward momentum.

**Challenges & Weaknesses:**

1. **Difficulty relaxing**: When every moment feels priceless, unwinding without a plan or tangible outcome might become increasingly hard.
2. **Interpersonal limitations**: Close relationships may struggle due to expectations for efficiency and prompt responses. Social interactions risk being transactional, sacrificing emotional depth.
3. **Burnout vulnerability**: Failing to balance relentless productivity with rejuvenation could lead to exhaustion.""")

    text_mutator = EntropyWordMutator()

    start = time.time()

    # df = pd.read_csv('./data/WQE_adaptive/dev.csv')

    # mutations_file_path = './inputs/word_mutator/test_2new.csv'

    mutated_text = text

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

    # for idx, row in df.head(50).iterrows():
    #     mutated_text = row['text']

    #     words, punct = text_mutator.get_words(mutated_text)

    #     log.info(f"Words: {words}")
    #     log.info(f"Punct: {punct}")

    #     # for _ in range(20):
    #     mutated_text = text_mutator.mutate(mutated_text)
    #     delta = time.time() - start

    #     log.info(f"Original text: {text}")
    #     log.info(f"Mutated text: {mutated_text}")
    #     log.info(f"Original == Mutated: {text == mutated_text}")
    #     # log.info(f"Diff: {text_mutator.diff(text, mutated_text)}")
    #     log.info(f"Time taken: {delta}")

    #     stats = [{'id': row.id, 'text': row.text, 'zscore' : row.zscore, 'watermarking_scheme': row.watermarking_scheme, 'model': row.model, 'gen_time': row.time, 'mutation_time': delta, 'mutated_text': mutated_text}]
    #     save_to_csv(stats, mutations_file_path, rewrite=False)

if __name__ == "__main__":
    test()