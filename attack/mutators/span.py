import time
import torch
import transformers
import re
import math
from tqdm import tqdm
import numpy as np
import difflib
import hydra
import logging
from nltk.tokenize import word_tokenize

log = logging.getLogger(__name__)

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def extract_fills(texts):
    # log.info(f"extract_fills (texts in): {texts}")
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    pattern = re.compile(r"<extra_id_\d+>")
    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    # log.info(f"extract_fills (texts out): {extracted_fills}")
    return extracted_fills

def join_tokens(tokens):
    joined = " ".join(tokens).replace(" \n", "\n")
    # Remove spaces before certain punctuation marks
    joined = re.sub(r'\s([,.;!?])', r'\1', joined)
    return joined

def apply_extracted_fills(masked_texts, extracted_fills):
    # log.info(f"apply_extracted_fills (masked_texts in): {masked_texts}")
    # log.info(f"apply_extracted_fills (extracted_fills in): {extracted_fills}")
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.replace('\n', ' \n').split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [join_tokens(x) for x in tokens]
    # log.info(f"apply_extracted_fills (texts out): {texts}")lo
    return texts

class Args:
    def __init__(self):
        self.cache_dir = './.cache'
        self.dist_alpha = 0.05
        self.checkpoint_alpha = 0.05
        self.check_quality = True
        self.watermark_scheme = "umd"
        self.tie_threshold = 0.001
        self.repetition_penalty = 1.1
        self.mask_top_p = 0.95
        self.n_spans = 1
        self.span_len = 6
        self.gen_len = 200
        self.step_T = 400
        self.mask_filling_model_name = "google/t5-v1_1-xl" # "google/flan-t5-xl"
        self.chunk_size = 20
        self.int8 = False
        self.half = False
        self.buffer_size = 1
        self.random_fills = False
        self.verbose = False

class SpanMutator:
    def __init__(self) -> None:
        self.n_resample = 5
        self.args = Args()
        self.verbose = self.args.verbose
        self.mask_filling_model_name = self.args.mask_filling_model_name
        self.n_positions = 1024

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 't5' in self.args.mask_filling_model_name:
            self.mask_model = self.load_mask_model()
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.mask_filling_model_name, model_max_length=self.n_positions)

        self.cached_replaced_tokens = set()
        self.original_tokens = set()

    def load_mask_model(self):
        int8_kwargs = {}
        half_kwargs = {}
        if self.args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif self.args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        if self.verbose: log.info(f'Loading mask filling model {self.args.mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.args.mask_filling_model_name, 
            cache_dir="/data2/.shared_models",
            **int8_kwargs, 
            **half_kwargs,
        )
        try:
            self.n_positions = self.mask_model.config.n_positions
        except AttributeError:
            self.n_positions = 1024
        if self.verbose: log.info('MOVING MASK MODEL TO GPU...')
        start = time.time()
        if not self.args.random_fills and not self.args.int8:
            mask_model.to(self.device)
        if self.verbose: log.info(f'DONE ({time.time() - start:.2f}s)')
        return mask_model

    def tokenize_and_mask(self, text, span_len, pct, ceil_pct=False):
        # log.info(f"tokenize_and_mask (text in): {text}")
        tokens = text.replace('\n', ' \n').split(' ')
        mask_string = '<<<mask>>>'
        # only mask one span
        n_spans = self.args.n_spans

        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start_pos = 0 # only need to prevent moefiying the instruction for chat models as they repeat Q-A.
            # if self.args.dataset == "c4_realnews":
            #     start_pos = 0
            # else:
            # # chat models might repeat the Q:.... A: prompt. So avoid query being perturbed.
            #     start_pos = len(self.prefix.replace('\n', ' \n').split(' '))
            if self.verbose: log.info(f"start_pos: {start_pos}")
            if self.verbose: log.info(f"len(tokens): {len(tokens)}")
            if self.verbose: log.info(f"span_len: {span_len}")
            start = np.random.randint(start_pos, len(tokens) - span_len)

            end = start + span_len
            search_start = max(0, start - self.args.buffer_size)
            search_end = min(len(tokens), end + self.args.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                # record/remove already masked tokens
                masked_tokens = set(tokens[start:end])
                if len(masked_tokens) > 1:
                    self.cached_replaced_tokens |= masked_tokens
                tokens[start:end] = [mask_string]
                n_masks += 1

        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        
        text = ' '.join(tokens).replace(" \n", "\n")
				
        if self.verbose: log.info(f"tokenize_and_mask (text out): {text}")
        return text

    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
        if self.verbose: log.info(f"replace_masks (texts in): {texts}")
        n_expected = count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        if self.verbose: log.info(f"replace_masks (stop_id): {stop_id}")
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        if self.verbose: log.info(f"replace_masks (tokens): {tokens}")

        min_len = int(np.ceil(self.args.span_len * self.args.n_spans * 1.5))
        max_len = int(self.args.span_len*self.args.n_spans*2)
        if self.verbose: log.info(f"min length: {min_len} max length: {max_len}")
        outputs = self.mask_model.generate(**tokens,
                                           max_length=max_len,
                                           min_length=min_len,
                                           do_sample=True,
                                           top_p=self.args.mask_top_p,
                                           num_return_sequences=1,
                                           repetition_penalty=self.args.repetition_penalty,
                                           eos_token_id=stop_id)  # 500 max, 150
        texts = self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        if self.verbose: log.info(f"replace_masks (texts out): {texts}")
        return texts

    def perturb_texts_(self, texts, span_len, pct, ceil_pct=False):
        if self.verbose: log.info(f"perturb_texts_ (texts in): {texts}")
        masked_texts = []
        for x in texts:
            masked_texts.append(self.tokenize_and_mask(x, span_len, pct, ceil_pct))

        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            # log.warn(f'{len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_len, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1

        if self.verbose: log.info(f"perturb_texts_ (texts out): {perturbed_texts}")
        return perturbed_texts

    def perturb_texts_t5(self, texts, span_len, pct, k=5, ceil_pct=False):
        if self.verbose: log.info(f"perturb_texts_t5 (texts in): {texts}")
        chunk_size = self.args.chunk_size
        if '11b' in self.args.mask_filling_model_name:
            chunk_size //= 2

        outputs = []
        # set chunk_size as 1 to help make sure each original token is replaced.
        for i in range(0, len(texts), chunk_size):
        # for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
            outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], span_len, pct, ceil_pct=ceil_pct))

        # log.info(f"perturb_texts_t5 (texts out): {outputs}")
        return outputs

    def paraphrase(self, texts, k=5):
        return self.perturb_texts_t5(texts, span_len=self.args.span_len, pct=0.2, k=k, ceil_pct=False)

    def mutate(self, text, k=5):
        return self.paraphrase([text], k=k)[0]


class Span2Mutator(SpanMutator):
    def __init__(self):
        super().__init__()
        self.args.span_len = 2


class Span3Mutator(SpanMutator):
    def __init__(self):
        super().__init__()
        self.args.span_len = 3


class Span4Mutator(SpanMutator):
    def __init__(self):
        super().__init__()
        self.args.span_len = 4



if __name__ == "__main__":

    def test():
        import time
        from utils import diff
        import pandas as pd

        print(f"Starting mutation...")

        dataset = pd.read_csv("./data/WQE/dev.csv")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        n=20
        avg_time = 0
        dataset = dataset.head(n) 
        
        text_mutator = SpanMutator()
        
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