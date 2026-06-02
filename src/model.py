from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from attention import CausalSelfAttention
from rmsnorm import RMSNorm
from swiglu import SwiGLU


@dataclass
class LlamaConfig:
    name: str = "llama_inspired_100m"
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 10
    n_embd: int = 768
    n_head: int = 12
    head_dim: int | None = None
    n_kv_head: int | None = None
    ffn_hidden_size: int = 2048
    dropout: float = 0.0
    bias: bool = False
    norm: str = "rmsnorm"
    activation: str = "swiglu"
    positional_encoding: str = "rope"
    tie_weights: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.n_embd // self.n_head
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_head * self.head_dim != self.n_embd:
            raise ValueError("n_head * head_dim must equal n_embd")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

    @classmethod
    def from_dict(cls, values: dict) -> "LlamaConfig":
        allowed = cls.__dataclass_fields__.keys()
        filtered = {key: value for key, value in values.items() if key in allowed}
        return cls(**filtered)


class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config.n_embd, config.ffn_hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")

        x = self.tok_embeddings(idx)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
        return idx

