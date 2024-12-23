from omegaconf import OmegaConf

KGW_cfg_dict = {
    'generator_args': {
        'model_name_or_path': 'meta-llama/Meta-Llama-3.1-70B-Instruct',  # Path to the generation model
        'revision': 'main',
        'model_cache_dir': '/data2/.shared_models/',  # Directory for cached models
        'device_map': 'auto',
        'trust_remote_code': True,
        'max_new_tokens': 1024,  # Maximum tokens to generate
        'min_new_tokens': 128,   # Minimum tokens to generate
        'do_sample': True,       # Whether to sample or use greedy decoding
        'no_repeat_ngram_size': 0,
        'temperature': 1.0,      # Sampling temperature
        'top_p': 0.95,           # Nucleus sampling parameter
        'top_k': 40,             # Top-K sampling parameter
        'repetition_penalty': 1.1,  # Penalty for repeated tokens
        'watermark_score_threshold': 5.0,
        'diversity_penalty': 0.0
    },
    'watermark_args': {
        'name': 'umd',  # Type of watermark
        'gamma': 0.25,   # Watermark strength parameter
        'delta': 2.0,    # Watermark behavior parameter
        'seeding_scheme': 'selfhash',
        'ignore_repeated_ngrams': True,
        'normalizers': [],  # List of normalization functions
        'z_threshold': 0.5,
        'device': 'cuda',   # Device for watermarking operations
        'only_detect': False  # Whether to only detect watermark without embedding
    }
}

KGW_cfg = OmegaConf.create(KGW_cfg_dict)