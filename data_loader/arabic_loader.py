# data_loader/arabic_loader.py
# Arabic Wikipedia data loader for pretraining

from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def load_arabic_data(max_samples=50000, max_length=512):
    """
    Load Arabic Wikipedia dataset.
    Uses AraBERT tokenizer -- handles Arabic script correctly.
    """
    print("Loading Arabic Wikipedia dataset...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ar",
        split="train",
        trust_remote_code=True
    )

    # Take a subset for Colab
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} Arabic articles")

    # Use AraBERT tokenizer -- handles Arabic RTL correctly
    print("Loading Arabic tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )

    print("Tokenizing Arabic text...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    tokenized.set_format("torch")
    print(f"Arabic dataset ready: {len(tokenized)} samples")

    return tokenized, tokenizer
