from omegaconf import OmegaConf

adaptive_cfg_dict = {
    'generator_args': {
        'model_name_or_path': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'revision': 'main',
        'model_cache_dir': '/data2/.shared_models/',
        'device_map': 'auto',
        'trust_remote_code': True,
        'max_new_tokens': 1024,
        'min_new_tokens': 128,
        'do_sample': True,
        'no_repeat_ngram_size': 0,
        'temperature': 1.0,
        'top_p': 0.9,
        'top_k': 50,
        'repetition_penalty': 1.1,
        'watermark_score_threshold': 5.0,
        'diversity_penalty': 0.0
    },
    'watermark_args': {
        'name': 'adaptive',
        'gamma': 0.25,
        'delta': 1.5,
        'seeding_scheme': 'selfhash',
        'ignore_repeated_ngrams': True,
        'normalizers': [],
        'z_threshold': 0.5,
        'device': 'auto',
        'only_detect': False,
        'measure_model_name': 'EleutherAI/gpt-neo-2.7B',
        'embedding_model_name': 'sentence-transformers/all-mpnet-base-v2',
        'delta_0': 1.0,
        'alpha': 2.0,
        'no_repeat_ngram_size': 0,
        'secret_string': 'The quick brown fox jumps over the lazy dog',
        'measure_threshold': 50,
        'detection_threshold': 95.0
    }
}

adaptive_cfg = OmegaConf.create(adaptive_cfg_dict)