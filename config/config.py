# config/config.py
# Simplified pretraining config — no dataclasses, no CLI loader

# ── Model sizes ──────────────────────────────────────────────
MODEL_CONFIGS = {
    "13M": {
        "vocab_size":  50257,
        "d_model":     256,
        "n_heads":     8,
        "n_layers":    6,
        "d_ff":        1024,
        "max_seq_len": 512,
        "dropout":     0.1,
    },
    "30M": {
        "vocab_size":  50257,
        "d_model":     512,
        "n_heads":     8,
        "n_layers":    8,
        "d_ff":        2048,
        "max_seq_len": 512,
        "dropout":     0.1,
    },
    "60M": {
        "vocab_size":  50257,
        "d_model":     768,
        "n_heads":     12,
        "n_layers":    12,
        "d_ff":        3072,
        "max_seq_len": 512,
        "dropout":     0.1,
    },
}

# ── Training ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    "batch_size":       32,
    "learning_rate":    3e-4,
    "max_steps":        10000,
    "warmup_steps":     500,
    "grad_clip":        1.0,
    "eval_interval":    500,
    "save_interval":    1000,
    "device":           "cuda",   # auto-detected in notebooks
}

# ── Data ──────────────────────────────────────────────────────
DATA_CONFIG = {
    "english": {
        "dataset_name": "openwebtext",
        "tokenizer":    "gpt2",
    },
    "arabic": {
        "dataset_name": "wikipedia",
        "dataset_lang": "20231101.ar",
        "tokenizer":    "aubmindlab/bert-base-arabertv2",
    },
}

# ── Default Config for Standalone Scripts ─────────────────────
import torch
default_config = {
    "n_head":           8,
    "n_embed":          512,
    "context_length":   512,
    "vocab_size":       50257,
    "n_blocks":         8,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
}
