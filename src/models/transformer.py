from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from src.models.transformer_block import Block

class Transformer(nn.Module):
    """
    The main Transformer model.

    This class combines token and position embeddings with a sequence of Transformer blocks
    and a final linear layer for language modeling.

    Args:
        n_head (int): The number of attention heads in each transformer block.
        n_embed (int): The dimensionality of the embedding space.
        context_length (int): The maximum length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        N_BLOCKS (int): The number of transformer blocks in the model.
    """
    def __init__(self, 
                 n_head: int = None, 
                 n_embed: int = None, 
                 context_length: int = None, 
                 vocab_size: int = None, 
                 N_BLOCKS: int = None,
                 d_model: int = None,
                 n_heads: int = None,
                 n_layers: int = None,
                 max_seq_len: int = None,
                 **kwargs) -> None:
        """
        Initializes the Transformer model, supporting both original parameter names
        and simplified config names.
        """
        super().__init__()
        
        # Map parameters if config-style naming is used
        if n_embed is None:
            n_embed = d_model
        if n_head is None:
            n_head = n_heads
        if N_BLOCKS is None:
            N_BLOCKS = n_layers
        if context_length is None:
            context_length = max_seq_len
            
        # Validate that all required properties are set
        if n_head is None or n_embed is None or context_length is None or vocab_size is None or N_BLOCKS is None:
            raise ValueError(
                f"Missing required parameters to initialize Transformer. "
                f"Got: n_head={n_head}, n_embed={n_embed}, context_length={context_length}, "
                f"vocab_size={vocab_size}, N_BLOCKS={N_BLOCKS}"
            )
            
        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        self.gradient_checkpointing = False
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        self.attn_blocks = nn.ModuleList([Block(n_head, n_embed, context_length) for _ in range(N_BLOCKS)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('pos_idxs', torch.arange(context_length))

    def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Combines token and position embeddings.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Sum of token and position embeddings.
        """
        B, T = idx.shape
        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        return tok_embedding + pos_embedding

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and return the final hidden states AFTER the final layer norm.

        This is exactly the tensor that ``lm_head`` consumes, so it is the right
        representation for auxiliary heads added during post-training (a scalar value
        head for PPO, a scalar reward head for the reward model). Keeping it as a
        separate method lets those heads reuse the backbone without duplicating the
        forward logic or rewriting ``forward``.

        Args:
            idx (torch.Tensor): Input token indices, shape (B, T).

        Returns:
            torch.Tensor: Final hidden states, shape (B, T, n_embed).
        """
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks:
            if self.gradient_checkpointing and self.training:
                # Recompute block activations in backward instead of storing them,
                # trading compute for a large activation-memory saving. use_reentrant=False
                # is the modern, correct variant.
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.layer_norm(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Transformer.

        Args:
            idx (torch.Tensor): Input token indices.
            targets (torch.Tensor, optional): Target token indices for loss calculation. Defaults to None.

        Returns:
            torch.Tensor | tuple: Logits tensor (if targets is None) or a tuple of (logits, loss).
        """
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T).long()
            loss = F.cross_entropy(flat_logits, targets)
            return logits, loss
        return logits

    def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass focusing on the embedding and attention blocks.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            tuple: Output after attention blocks and the residual.
        """
        x = self._pre_attn_pass(idx)
        residual = x
        for block in self.attn_blocks:
            x, residual = block.forward_embedding(x)
        return x, residual

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens given a starting sequence.

        Args:
            idx (torch.Tensor): Initial sequence of token indices.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: The extended sequence of tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    vocab_size = 100
    embedding_dim = 32
    num_heads = 4
    num_blocks = 2
    context_len = 5
    input_indices = torch.randint(0, vocab_size, (batch_size, sequence_length))

    transformer_model = Transformer(n_head=num_heads, n_embed=embedding_dim, context_length=context_len, vocab_size=vocab_size, N_BLOCKS=num_blocks)
    logits, loss = transformer_model(input_indices, targets=input_indices) # Using input as target for simplicity

    print("Transformer Logits Shape:", logits.shape)
    print("Transformer Loss:", loss)

    # Example of generating tokens
    start_indices = input_indices[:, :1]  # Take the first token of each sequence as start
    generated_tokens = transformer_model.generate(start_indices, max_new_tokens=5)
    print("Generated Tokens Shape:", generated_tokens.shape)
