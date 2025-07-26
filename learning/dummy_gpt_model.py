import torch
from importlib.metadata import version
import torch.nn as nn

print("matplotlib version:", version("matplotlib"))
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
    	print("in_idx:", in_idx)
    	batch_size, seq_len = in_idx.shape
    	print("in_idx.shape:", in_idx.shape)
    	tok_embeds = self.tok_emb(in_idx)
    	pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    	print("tok_embeds:", tok_embeds)
    	print("tok_embeds.shape:", tok_embeds.shape)
    	print("pos_embeds:", pos_embeds)
    	print("pos_embeds.shape:", pos_embeds.shape)
    	x = tok_embeds + pos_embeds
    	print("x.shape before:", x.shape)
    	print("x:", x)
    	x = self.drop_emb(x)
    	x = self.trf_blocks(x)
    	x = self.final_norm(x)
    	print("x.shape after:", x.shape)
    	logits = self.out_head(x)
    	print("logits.shape:", logits.shape)
    	print("logits:", logits)
    	return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print("batch:", batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
print("logits shape:", logits.shape)
print(logits)


