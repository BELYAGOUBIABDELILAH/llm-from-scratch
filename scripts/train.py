#!/usr/bin/env python3
"""
Simple pretraining script for custom Transformer models from scratch.
"""

import os
import time
import argparse
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.models.transformer import Transformer
from config.config import MODEL_CONFIGS

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model from scratch.")
    parser.add_argument("--model_size", type=str, default="13M", choices=["13M", "30M", "60M"], help="Model parameter configuration to train.")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--output_path", type=str, default="./checkpoints/model_checkpoint.pt", help="Path to save model checkpoint.")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Load config and model
    cfg = MODEL_CONFIGS[args.model_size]
    model = Transformer(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded {args.model_size} model configuration ({n_params:,} parameters)")
    
    # 2. Load dataset
    print("Loading Salesforce/wikitext dataset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        
    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    
    # 3. Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(tokenized, batch_size=args.batch_size, shuffle=True)
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    losses = []
    print(f"Starting training for {args.steps} steps...")
    model.train()
    
    t0 = time.time()
    for step, batch in enumerate(tqdm(loader, total=args.steps)):
        if step >= args.steps:
            break
            
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        logits = model(input_ids)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, cfg["vocab_size"]),
            shift_labels.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")
            
    training_time = time.time() - t0
    perplexity = torch.exp(torch.tensor(losses[-50:]).mean()).item()
    print(f"Training completed in {training_time:.1f}s | Final Loss: {sum(losses[-50:])/50:.4f} | Perplexity: {perplexity:.2f}")
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config':           cfg,
        'final_loss':       round(sum(losses[-50:]) / 50, 4),
        'perplexity':       round(perplexity, 2),
        'training_time':    round(training_time, 1),
    }, args.output_path)
    print(f"Saved checkpoint to: {args.output_path}")

if __name__ == "__main__":
    main()
