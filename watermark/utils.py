from transformers import AutoModelForCausalLM, AutoTokenizer

def init_model(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        revision=cfg.revision,
        cache_dir=cfg.model_cache_dir,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code)
    return model

def init_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.generator_args.model_name_or_path, 
        use_fast=True, 
        cache_dir=cfg.model_cache_dir)
    
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer
