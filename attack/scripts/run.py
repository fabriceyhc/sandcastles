import os
import traceback
import pandas as pd
from attack.mutators import (
    SentenceMutator, SpanMutator, WordMutator, EntropyWordMutator, 
    DocumentMutator, Document1StepMutator, Document2StepMutator
)
from attack.oracles import DiffOracle, InternLMOracle
from attack.attack import Attack

import hydra
import os
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

@hydra.main(version_base=None, config_path="./config", config_name="attack")
def main(cfg):

    cfg.attack.check_watermark = False

    datasets = [
<<<<<<< HEAD:attack/scripts/run.py
        "GPT4o_unwatermarked",
=======
        # "GPT4o_unwatermarked",
>>>>>>> 63b8aaf3b937eb76d557f744e275dcb09e415cf9:attack/run.py
        # "Adaptive",
        # "EXP",
        "KGW",
    ]

    mutators = [
        # EntropyWordMutator, 
        # WordMutator,
        # SpanMutator,
<<<<<<< HEAD:attack/scripts/run.py
        SentenceMutator,
=======
        # SentenceMutator,
>>>>>>> 63b8aaf3b937eb76d557f744e275dcb09e415cf9:attack/run.py
        Document1StepMutator,
        Document2StepMutator,
        DocumentMutator,
    ]

    # Initialize Quality Oracle
    oracle = InternLMOracle()

    for d in datasets:

        # Load data
        data = pd.read_csv(f"./data/texts/entropy_control_{d}.csv")

        for mutator in mutators:
            # Initialize Mutator
            log.info("Initializing mutator...")
            mutator = mutator()
            log.info("Mutator initialized.")

            # mInitialize Attacker
            o_str = oracle.__class__.__name__
            m_str = mutator.__class__.__name__
            cfg.attack.compare_against_original = True

            if m_str == "WordMutator":
                cfg.attack.max_steps = 1000
            if m_str == "EntropyWordMutator":
                cfg.attack.max_steps = 1000
            if m_str == "SpanMutator":
                cfg.attack.max_steps = 250
            if "Sentence" in m_str:
                cfg.attack.max_steps = 150
            if "Document" in m_str:
                cfg.attack.max_steps = 100

            cfg.attack.log_csv_path = f"./attack/traces/{o_str}_{d}_{m_str}_n-steps={cfg.attack.max_steps}_attack_results.csv"

            if os.path.exists(cfg.attack.log_csv_path):
                log.info(f"skipping this attack configuration: {cfg.attack.log_csv_path}")
                continue

            log.info(f"Initializing attacker...")
            attacker = Attack(cfg, mutator, oracle)

            # Step 6: Attack each row in dataset
            for benchmark_id, row in data.iterrows():
                try:
                    log.info(f"Attacking Row: {row}")
                    attacked_text = attacker.attack(row['prompt'], row['text'])
                    log.info(f"Attacked Text: {attacked_text}")
                except Exception:
                    log.info(traceback.format_exc())

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m attack.scripts.run

    main()
