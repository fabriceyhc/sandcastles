from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig

from watermark.config.KGW import KGW_cfg
from watermark.adaptive import AdaptiveWatermarker
from watermark.config.Adaptive import adaptive_cfg

def get_watermark(watermark):
    # TODO: If EXP is successful, adapt this to work with EXP as well.
    if watermark == "adaptive":
        return AdaptiveWatermarker(adaptive_cfg)
    
    model = AutoModelForCausalLM.from_pretrained(KGW_cfg.generator_args.model_name_or_path, device_map='auto', cache_dir=KGW_cfg.generator_args.model_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        KGW_cfg.generator_args.model_name_or_path, 
        use_fast=True, 
        cache_dir=KGW_cfg.generator_args.model_cache_dir) 
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Transformers config
    transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=tokenizer.vocab_size,
                                            max_new_tokens=KGW_cfg.generator_args.max_new_tokens,
                                            min_length=KGW_cfg.generator_args.min_new_tokens,
                                            do_sample=KGW_cfg.generator_args.do_sample,
                                            no_repeat_ngram_size=KGW_cfg.generator_args.no_repeat_ngram_size)
    
    # Load watermark algorithm
    myWatermark = AutoWatermark.load(watermark, 
                                    algorithm_config=f'../watermark/config/{watermark}.json',
                                    transformers_config=transformers_config)

    return myWatermark
    