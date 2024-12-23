from abc import ABC, abstractmethod
import torch
import logging

from watermark.utils import init_model, init_tokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Watermarker(ABC):
    def __init__(self, cfg, model = None, tokenizer = None, n_attempts=1, only_detect=True):
        self.cfg = cfg # the entire config is passed, since we want to look at the generation_args as well
        self.n_attempts = n_attempts
        self.model = model
        self.tokenizer = tokenizer
        self.only_detect = only_detect
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        log.info(f"Using device: {self.device}")

        if self.tokenizer is None:
            self.tokenizer = init_tokenizer(self.cfg)

        if not self.only_detect and self.model is None:            
            self.model = init_model(self.cfg)

        key_max_length = "max_length" if "Llama-3" in self.cfg.generator_args.model_name_or_path else "max_new_tokens"
        key_min_length = "min_length" if "Llama-3" in self.cfg.generator_args.model_name_or_path else "min_new_tokens"

        self.generator_kwargs = {
            key_max_length: self.cfg.generator_args.max_new_tokens,
            key_min_length: self.cfg.generator_args.min_new_tokens,
            "do_sample": self.cfg.generator_args.do_sample,
            "temperature": self.cfg.generator_args.temperature,
            "top_p": self.cfg.generator_args.top_p,
            "top_k": self.cfg.generator_args.top_k,
            "repetition_penalty": self.cfg.generator_args.repetition_penalty
        }

        self._setup_watermark_components()

    @abstractmethod
    def _setup_watermark_components(self):
        pass

    @abstractmethod
    def generate_watermarked_outputs(self, prompt):
        pass

    def generate(self, prompt, **kwargs):
        n_attempts = 0
        while n_attempts < self.n_attempts:
            completion = self.generate_watermarked_outputs(prompt, **kwargs)

            log.info(f"Received watermarked text: {completion}")

            if not self.cfg.is_completion:
                completion = completion.replace(prompt, '', 1).strip()

            # Check if watermark succeeded
            if self.cfg.watermark_args.name == "adaptive":
                is_detected = self.detect(completion)
            else:
                is_detected, _ = self.detect(completion)
            if is_detected:
                return completion
            else:
                log.info("Failed to watermark, trying again...")
                n_attempts += 1

        return None

    @abstractmethod
    def detect(self, completion):
        pass




