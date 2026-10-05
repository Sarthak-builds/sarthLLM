SARTHLLM_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,  # max tokens per sequence; must equal the dataset's max_length
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

cfg = SARTHLLM_CONFIG