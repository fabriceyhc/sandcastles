import os
from huggingface_hub import snapshot_download

models = [
    ("bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF", "Q8_0"),
    ("MaziyarPanahi/Llama-3.3-70B-Instruct-GGUF",         "Q8_0"),
    ("MaziyarPanahi/DeepSeek-V2.5-GGUF",                  "Q4_K_M"),
    ("unsloth/DeepSeek-V3-GGUF",                          "Q2_K_XS"),
]

# Common metadata/config files we usually want (adjust as needed).
# You could add "tokenizer.json" if it exists in some repos, or "model_config.json" etc.
COMMON_FILES = [
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "config.json",
]

for repo_id, quant_level in models:
    print(f"Downloading model for {repo_id} with quant level '{quant_level}'...")

    # Build allow_patterns so we only grab:
    #   1. Files containing the quant level (in any extension).
    #   2. The usual config/tokenizer files.
    # Adjust or add patterns to match your exact naming.
    # For instance, some repos might have "q8_0" in the filename vs. "Q8_0", etc.
    allow_patterns = [f"*{quant_level}*"] + COMMON_FILES

    # Local directory for the downloaded files
    local_dir = f"/data2/.shared_models/llama.cpp_models/{repo_id.split('/')[-1]}"

    # Download only the allowed patterns
    downloaded_folder = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,         # key line: only these patterns
        local_dir_use_symlinks=False,          # store real copies, not symlinks
        # revision="main"  # or a specific branch/tag/commit if needed
    )

    print(f"  -> Files downloaded to: {downloaded_folder}\n")
