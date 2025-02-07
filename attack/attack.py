import logging
import time
import os
import pandas as pd
from tqdm import tqdm
import copy
from attack.utils import (
    save_to_csv,
    length_diff_exceeds_percentage,
    count_num_of_words
)

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class Attack:
    def __init__(self, cfg, mutator, quality_oracle=None, watermarker=None):
        self.cfg = cfg
        self.watermarker = watermarker
        self.mutator = mutator
        self.quality_oracle = quality_oracle
        self.results = []
        self.mutated_texts = []
        self.backtrack_patience = 0
        self.patience = 0
        self.max_mutation_achieved = 0
        self.base_step_metadata = {
            "step_num": -1,
            "mutation_num": -1,
            "prompt": "",
            "current_text": "",
            "mutated_text": "",
            "current_text_len": -1,
            "mutated_text_len": -1,
            "length_issue": False,
            "quality_analysis": {},
            "quality_preserved": False,
            "watermark_detected": False,
            "watermark_score": None,
            "backtrack": False,
            "total_time": "",
            "mutator_time": "",
            "oracle_time": "",
        }

        self.use_max_steps = self.cfg.attack.use_max_steps
        self.num_steps = self.cfg.attack.max_steps if self.use_max_steps else self.cfg.attack.target_mutations

        if self.cfg.attack.check_quality:
            assert quality_oracle is not None, "Quality oracle required!"
        if self.cfg.attack.check_watermark:
            assert watermarker is not None, "Watermark detector required!"

    def _reset(self):
        self.results = []
        self.mutated_texts = []
        self.backtrack_patience = 0
        self.patience = 0
        self.max_mutation_achieved = 0

    def _load_existing_state(self, prompt, original_text):
        try:
            if not os.path.exists(self.cfg.attack.log_csv_path):
                return None

            df = pd.read_csv(self.cfg.attack.log_csv_path)
            initial_step = df[(df['prompt'] == prompt) & 
                            (df['step_num'] == -1) &
                            (df['current_text'] == original_text)]

            if not initial_step.empty:
                attack_steps = df[(df['prompt'] == prompt) & 
                                ((df['current_text'] == original_text) | 
                                (df['step_num'] != -1))].sort_values('step_num')

                state = {
                    'mutated_texts': [original_text],
                    'successful_mutation_count': 0,
                    'step_num': -1,
                    'backtrack_patience': 0,
                    'patience': 0,
                    'max_mutation_achieved': 0
                }

                for _, step in attack_steps.iterrows():
                    if step['step_num'] == -1:
                        continue

                    state['step_num'] = step['step_num']
                    
                    if step['backtrack']:
                        if len(state['mutated_texts']) > 1:
                            state['mutated_texts'].pop()
                            state['successful_mutation_count'] -= 1
                        state['backtrack_patience'] = 0
                    
                    if step['mutation_num'] > state['successful_mutation_count']:
                        state['mutated_texts'].append(step['mutated_text'])
                        state['successful_mutation_count'] = step['mutation_num']
                        if state['successful_mutation_count'] > state['max_mutation_achieved']:
                            state['max_mutation_achieved'] = state['successful_mutation_count']
                            state['patience'] = 0
                        state['backtrack_patience'] = 0
                    else:
                        state['backtrack_patience'] += 1
                        state['patience'] += 1

                return state
        except Exception as e:
            log.error(f"Error loading state: {str(e)}")
        return None

    def backtrack(self):
        self.backtrack_patience = 0
        self.step_data.update({"backtrack": True})
        if len(self.mutated_texts) > 1:
            del self.mutated_texts[-1]
            self.successful_mutation_count -= 1
            self.current_text = self.mutated_texts[-1]

    def length_check(self):
        self.length_issue, self.original_len, self.mutated_len = length_diff_exceeds_percentage(
            text1=self.original_text,
            text2=self.mutated_text,
            percentage=self.cfg.attack.length_variance
        )
        self.current_text_len = count_num_of_words(self.current_text)
        self.step_data.update({
            "current_text_len": self.current_text_len,
            "mutated_text_len": self.mutated_len,
            "length_issue": self.length_issue
        })

    def append_and_save_step_data(self):
        self.step_data.update({"total_time": time.time() - self.start_time})
        self.results.append(self.step_data)
        save_to_csv([self.step_data], self.cfg.attack.log_csv_path)
        self.step_data.update({"backtrack": False})

    def check_watermark(self):
        watermark_detected, watermark_score = self.watermarker.detect(self.mutated_text)
        self.step_data.update({
            "watermark_detected": watermark_detected,
            "watermark_score": watermark_score
        })
        log.info(f"Watermark Score: {watermark_score}")
        self.append_and_save_step_data()
        return watermark_detected

    def is_attack_done(self):
        if self.use_max_steps:
            return self.step_num >= self.num_steps
        return self.successful_mutation_count >= self.num_steps

    def attack(self, prompt, watermarked_text):
        self._reset()
        self.original_text = watermarked_text
        self.current_text = watermarked_text

        # Try to load existing state
        existing_state = self._load_existing_state(prompt, watermarked_text)
        if existing_state:
            self.mutated_texts = existing_state['mutated_texts']
            self.successful_mutation_count = existing_state['successful_mutation_count']
            self.step_num = existing_state['step_num']
            self.backtrack_patience = existing_state['backtrack_patience']
            self.patience = existing_state['patience']
            self.max_mutation_achieved = existing_state['max_mutation_achieved']
            self.current_text = self.mutated_texts[-1]
            log.info(f"Resumed from step {self.step_num} with {self.successful_mutation_count} mutations")
            log.info(f"Last state: {existing_state}")
        else:
            self.mutated_texts = [self.original_text]
            self.successful_mutation_count = 0
            self.step_num = -1

        # Save initial state if resuming
        if self.step_num == -1:
            self.step_data = copy.copy(self.base_step_metadata)
            self.step_data.update({
                "prompt": prompt,
                "current_text": self.original_text,
                "watermark_detected": True,
                "quality_preserved": True
            })
            if self.cfg.attack.check_watermark:
                watermark_detected, watermark_score = self.watermarker.detect(self.original_text)
                if not watermark_detected:
                    raise ValueError("No watermark detected on input text!")
                self.step_data.update({
                    "watermark_detected": watermark_detected,
                    "watermark_score": watermark_score
                })
            save_to_csv([self.step_data], self.cfg.attack.log_csv_path)

        done = self.is_attack_done()
        initial_progress = self.step_num + 1 if self.use_max_steps else self.successful_mutation_count
        with tqdm(total=self.num_steps, initial=initial_progress) as pbar:
            while not done:
                self.start_time = time.time()
                
                if self.patience >= self.cfg.attack.patience:
                    log.error(f"Patience exceeded on mutation {self.successful_mutation_count}")
                    break

                self.step_num += 1
                pbar.set_description(f"Step {self.step_num}. Patience: {self.backtrack_patience};{self.patience}")
                self.step_data = copy.copy(self.base_step_metadata)
                self.step_data.update({
                    "step_num": self.step_num,
                    "mutation_num": self.successful_mutation_count,
                    "prompt": prompt,
                    "current_text": self.current_text
                })

                if self.backtrack_patience >= self.cfg.attack.backtrack_patience:
                    self.backtrack()

                # Mutation logic
                max_retries = self.cfg.attack.mutator_retries
                for retry_count in range(max_retries):
                    try:
                        mutate_start = time.time()
                        self.mutated_text = self.mutator.mutate(self.current_text)
                        self.step_data.update({
                            "mutated_text": self.mutated_text,
                            "mutator_time": time.time() - mutate_start
                        })
                        break
                    except Exception as e:
                        if retry_count == max_retries - 1:
                            raise
                        log.error(f"Mutation retry {retry_count+1} failed: {str(e)}")

                # Length check
                if self.cfg.attack.check_length:
                    self.length_check()
                    if self.length_issue:
                        self.backtrack_patience += 1
                        self.patience += 1
                        self.append_and_save_step_data()
                        if self.use_max_steps:
                            pbar.update(1)
                            done = self.is_attack_done()
                        continue

                # Quality check
                if self.cfg.attack.check_quality:
                    oracle_start = time.time()
                    quality_analysis = self.quality_oracle.is_quality_preserved(
                        prompt,
                        self.original_text if self.cfg.attack.compare_against_original else self.current_text,
                        self.mutated_text
                    )
                    self.step_data.update({
                        "quality_analysis": quality_analysis,
                        "quality_preserved": quality_analysis["quality_preserved"],
                        "oracle_time": time.time() - oracle_start
                    })
                    if not quality_analysis["quality_preserved"]:
                        self.backtrack_patience += 1
                        self.patience += 1
                        self.append_and_save_step_data()
                        if self.use_max_steps:
                            pbar.update(1)
                            done = self.is_attack_done()
                        continue

                # Update state
                self.current_text = self.mutated_text
                self.mutated_texts.append(self.mutated_text)

                # Watermark check
                if self.cfg.attack.check_watermark:
                    if not self.check_watermark():
                        log.info("Watermark removed!")
                        return self.mutated_text
                else:
                    self.append_and_save_step_data()

                self.successful_mutation_count += 1
                self.backtrack_patience = 0
                if self.successful_mutation_count > self.max_mutation_achieved:
                    self.max_mutation_achieved = self.successful_mutation_count
                    self.patience = 0

                pbar.update(1)
                done = self.is_attack_done()

        return self.current_text