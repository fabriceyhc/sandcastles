import logging
import torch
import openai
import json
from torch.nn import functional as F
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import random

from watermark.watermarker import Watermarker
from watermark.Adaptive.model import TransformModel
import textwrap

def vocabulary_mapping(vocab_size, sm_output_dim):
    return [random.randint(0, sm_output_dim-1) for _ in range(vocab_size)]

log = logging.getLogger(__name__)

class AdaptiveWatermarker(Watermarker):
    def __init__(self, cfg, model = None, tokenizer = None, n_attempts =1 , **kwargs):
        self.cfg = cfg
        super().__init__(cfg, model, tokenizer , n_attempts, **kwargs)

    def _load_mapping_list(self):
        if "opt" in self.cfg.generator_args.model_name_or_path:
            with open('/data2/.shared_models/adaptive_watermarker/mapping_opt.json', 'r') as f:
                self.mapping_list = json.load(f)
        elif "Llama-3" in self.cfg.generator_args.model_name_or_path:
            with open('/data2/.shared_models/adaptive_watermarker/mapping_llama31.json', 'r') as f:
                self.mapping_list = json.load(f)
        else:
            raise Exception("No mapping list provided for your model!")

    def _setup_watermark_components(self):
        """
        This function sets up the LLM we'll use for generating watermarked text.
        """

        # This is from their Github.
        self.measure_tokenizer = AutoTokenizer.from_pretrained(self.cfg.watermark_args.measure_model_name)
        self.measure_model = AutoModelForCausalLM.from_pretrained(self.cfg.watermark_args.measure_model_name, device_map='auto')
        self.measure_model.eval()

        # load semantic embedding model
        self.embedding_model = SentenceTransformer(self.cfg.watermark_args.embedding_model_name).to(self.device)
        self.embedding_model.eval()

        # load semantic mapping model
        self.transform_model = TransformModel()
        
        self.transform_model.load_state_dict(torch.load('/data2/.shared_models/adaptive_watermarker/transform_model.pth'))
        self.transform_model.to(self.device)
        self.transform_model.eval()

        # Load mapping list for the transform model
        self._load_mapping_list()
        

    def paraphrase(self, openai_api_key, input_text):
        openai.api_key = openai_api_key

        prompt_0 = 'You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n'
        prompt_1 = input_text
        prompt = prompt_0 + ' ' + prompt_1 + '.\n'

        try:
            response = openai.Completion.create(
                model = 'gpt-3.5-turbo-instruct',
                prompt = prompt,
                max_tokens = 300
            )
            output_text = response.choices[0].text.strip()

            return output_text
        
        except Exception as e:
            print('OpenAI API key is invalid!')
            print(input_text)
            return None

    def _calc_banned_ngram_tokens(self, prev_input_ids: torch.Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < no_repeat_ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - no_repeat_ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens

    def _postprocess_next_token_scores(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty, no_repeat_ngram_size):
        # _enforce_repetition_penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size * num_beams):
                for previous_token in set(prev_output_tokens[i].tolist()):
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if lprobs[i, previous_token] < 0:
                        lprobs[i, previous_token] *= repetition_penalty
                    else:
                        lprobs[i, previous_token] /= repetition_penalty
        
        # lower eos token prob to zero if min_length is not reached
        if prev_output_tokens.size(1) < self.cfg.generator_args.min_new_tokens:
            lprobs[:, self.tokenizer.eos_token_id] = -float("Inf")
        
        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = self._calc_banned_ngram_tokens(
                prev_output_tokens, num_batch_hypotheses, no_repeat_ngram_size, prev_output_tokens.size(1)
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                lprobs[i, banned_tokens] = -float("inf")

    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """ 
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
    
    def _stopping_criteria(self, ids, tokenizer):
        # stop_words = ["word.", "word!", "word?", "word...", "word;"]
        # stop_words_ids = [tokenizer.encode(stop_word, return_tensors='pt', add_special_tokens=False)[0][-1].to(self.device) for stop_word in stop_words]
        
        if ids[0][-1] == self.tokenizer.eos_token_id:
            return True

        # if ids[0][-1] in stop_words_ids:
        #     if len(ids[0]) > self.cfg.generator_args.min_new_tokens:
        #         return True
        return False

    def _next_token_entropy(self, input_text, model, tokenizer, device):
        if self.cfg.watermark_args.measure_model_name == "gpt2-large":
            input_ids = tokenizer.encode(input_text, truncation=True, max_length=1024, return_tensors='pt', add_special_tokens=False).to(device)
        else:
            input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).to(device)

        outputs = model(input_ids)
        probs = torch.nn.functional.softmax(outputs.logits[0, -1, :], dim=-1)
        mask = probs > 0
        entropy = -torch.sum(probs[mask] * torch.log(probs[mask]))
        return entropy

    def _bias_logits(self, logits, v_embedding, delta):
        logits = torch.mul(logits, (1 + delta*v_embedding))
        return logits

    def _watermarking(self, ids, logits, secret_string, measure_threshold):
        '''
        ids: Tensor, [[]]
        probs: Tensor, [[]]
        '''
        if len(ids[0]) <= measure_threshold:
            embedding = self.embedding_model.encode(secret_string, convert_to_tensor=True)
            t_embedding = self.transform_model(embedding).tolist()
            t_embedding = [1.0 if x>0.0 else 0.0 for x in t_embedding]
            v_embedding = torch.tensor([t_embedding[i] for i in self.mapping_list], device=self.device)
            logits[0] = self._bias_logits(logits[0], v_embedding, self.cfg.watermark_args.delta_0)
        elif len(ids[0]) > measure_threshold:
            measure_text = self.tokenizer.decode(ids[-1])
            measure_entropy = self._next_token_entropy(measure_text, self.measure_model, self.measure_tokenizer, self.device)
            log.info(f"Text: {measure_text}")
            log.info(f"Epntropy: {measure_entropy}")
            if measure_entropy >= self.cfg.watermark_args.alpha:
                embedding = self.embedding_model.encode(measure_text, convert_to_tensor=True)
                t_embedding = self.transform_model(embedding).tolist()   # torch([])
                t_embedding = [1.0 if x>0.0 else 0.0 for x in t_embedding]
                v_embedding = torch.tensor([t_embedding[i] for i in self.mapping_list], device=self.device)
                logits[0] = self._bias_logits(logits[0], v_embedding, self.cfg.watermark_args.delta)
        return logits

    def _find_repetitive_ngrams(self, text, n_gram, rep=2):
        # Tokenize the text
        tokens = word_tokenize(text)
        # Create n-grams from tokens
        ngrams = zip(*[tokens[i:] for i in range(n_gram)])
        ngram_counts = Counter(ngrams)
        # Find repetitive n-grams
        repetitive_ngrams = {ngram: count for ngram, count in ngram_counts.items() if count > rep}
        if len(repetitive_ngrams) > 0:
            return True
        else:
            return False
        
    # NOTE: Removed their unwatermarked text generation function since we can also do it on our own.
    # Adaptive watermark text generation
    def generate_watermarked_outputs(self, prompt):
        og_prompt = prompt
        if not self.cfg.is_completion:
            if "Llama" in self.model.config._name_or_path:
                prompt = textwrap.dedent(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful personal assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        output_ids = torch.tensor([[]], dtype=torch.int64, device=self.device)
        attn = torch.ones_like(input_ids)
        past = None

        length = len(input_ids[0])
        log.info(f"Length of the prompt in terms of tokens: {length}")
        # In order not to break Llama by generating more than 1024 tokens, include the prompt length in the range of the for loop.
        # It broke even when I substracted 150.

        for _ in range(self.cfg.generator_args.max_new_tokens - length):
            with torch.no_grad():
                if past:
                    output = self.model(input_ids[:,-1:], attention_mask=attn, past_key_values=past)
                else:
                    output = self.model(input_ids)
            
            logits = output.logits[:,-1, :]
            self._postprocess_next_token_scores(logits, 1, 1, output_ids, repetition_penalty=self.cfg.generator_args.repetition_penalty, no_repeat_ngram_size=self.cfg.watermark_args.no_repeat_ngram_size)
            logits = self._watermarking(output_ids, logits, self.cfg.watermark_args.secret_string, self.cfg.watermark_args.measure_threshold)   # watermarking
            logits = self._top_k_top_p_filtering(logits, top_k=self.cfg.generator_args.top_k, top_p=self.cfg.generator_args.top_p)   # top-k, top-p filtering
            probs = torch.nn.functional.softmax(logits, dim=-1)   # softmax
            next_id = torch.multinomial(probs, num_samples=1)   # sampling

            input_ids = torch.cat((input_ids, next_id), dim=-1)   # update input_ids
            output_ids = torch.cat((output_ids, next_id), dim=-1)   # update output_ids

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # stopping criteria
            stop = self._stopping_criteria(output_ids, self.tokenizer)
            if stop:
                output_text = self.tokenizer.decode(output_ids[0].tolist())
                return output_text
        
        output_text = self.tokenizer.decode(output_ids[0])
        
        return output_text
    
    def generate_adaptive_watermarked(self, prompt):
        count = 0
        resample = True
        while resample:
            output_text = self.generate_watermarked(prompt)
            resample = self._find_repetitive_ngrams(output_text, 4, 2)
            count += 1
            if count > 2:
                break
        return output_text

    def detect(self, text):
        watermark_ids = self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(self.device)
        
        e = self.embedding_model.encode(self.cfg.watermark_args.secret_string, convert_to_tensor=True, device=self.device)
        e = e.to(self.device)
        self.transform_model.to(self.device)
        te = self.transform_model(e).tolist()
        te = [1.0 if x>0.0 else 0.0 for x in te]
        ve = torch.tensor([te[i] for i in self.mapping_list], device=self.device)

        score = []
        for i in range(len(watermark_ids[0])):
            if i <= self.cfg.watermark_args.measure_threshold:
                s = ve[watermark_ids[0][i]]
                score.append(s)
            elif i > self.cfg.watermark_args.measure_threshold:
                measure_text = self.tokenizer.decode(watermark_ids[0][:i])
                measure_entropy = self._next_token_entropy(measure_text, self.measure_model, self.measure_tokenizer, self.device)
                if measure_entropy >= self.cfg.watermark_args.alpha:
                    e = self.embedding_model.encode(measure_text, convert_to_tensor=True, device=self.device)
                    te = self.transform_model(e).tolist()
                    te = [1.0 if x>0.0 else 0.0 for x in te]
                    ve = torch.tensor([te[i] for i in self.mapping_list], device=self.device)
                    s = ve[watermark_ids[0][i]]
                    score.append(s)
        
        normalized_score = sum(score)/len(score)
        normalized_score = normalized_score.item()

        normalized_score = normalized_score * 100

        is_detected = self.cfg.watermark_args.detection_threshold < normalized_score

        return is_detected, normalized_score
    