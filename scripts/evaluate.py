#!/usr/bin/env python3
"""
Simple evaluation script for LLM models.
Computes perplexity on a validation dataset.
"""

import torch
import argparse
from transformers import GPT2Tokenizer
from datasets import load_dataset
from src.models.transformer import Transformer
from config.config import MODEL_CONFIGS


def evaluate_model(model, dataloader, device):
    """Evaluate model and compute perplexity."""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            
            logits = model(input_ids)
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, model.vocab_size),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM model")
    parser.add_argument("--model_size", type=str, default="13M", choices=["13M", "30M", "60M"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load model
    config = MODEL_CONFIGS[args.model_size]
    model = Transformer(**config).to(args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"✅ Loaded model from {args.checkpoint}")
    
    # Load validation data
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    
    dataloader = torch.utils.data.DataLoader(tokenized, batch_size=args.batch_size)
    
    # Evaluate
    print("🔍 Evaluating model...")
    loss, perplexity = evaluate_model(model, dataloader, args.device)
    
    print(f"\n{'='*50}")
    print(f"📊 Evaluation Results")
    print(f"{'='*50}")
    print(f"Loss:       {loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
