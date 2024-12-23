from omegaconf import OmegaConf

cfg_dict = OmegaConf.to_container(cfg, resolve=True)
cfg_dict['generator_args']['top_k'] = 50
cfg_dict['generator_args']['top_p'] = 0.9
cfg_dict['generator_args']['max_new_tokens'] = 1024 # 285
cfg_dict['generator_args']['min_new_tokens'] = 128 # 215
cfg_dict['generator_args']['repetition_penalty'] = 1.1

cfg_dict['watermark_args']['name'] = "adaptive"
# cfg_dict['watermark_args']['measure_model_name'] = "gpt2-large"
cfg_dict['watermark_args']['measure_model_name'] = "EleutherAI/gpt-neo-2.7B"

cfg_dict['watermark_args']['embedding_model_name'] = "sentence-transformers/all-mpnet-base-v2"
cfg_dict['watermark_args']['delta'] = 1.5
cfg_dict['watermark_args']['delta_0'] = 1.0
cfg_dict['watermark_args']['alpha'] = 2.0
cfg_dict['watermark_args']['no_repeat_ngram_size'] = 0
cfg_dict['watermark_args']['secret_string'] = 'The quick brown fox jumps over the lazy dog'
cfg_dict['watermark_args']['measure_threshold'] = 50
cfg_dict['watermark_args']['detection_threshold'] = 95.0
cfg_dict['watermark_args']['device'] = 'auto'

cfg = OmegaConf.create(cfg_dict)

