from pathlib import Path

import numpy as np
import torch


class BinaryTokenDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token file not found: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.tokens) <= block_size + 1:
            raise ValueError(f"{self.path} must contain more than block_size + 1 tokens")

        ix = torch.randint(len(self.tokens) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy((self.tokens[int(i) : int(i) + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.tokens[int(i) + 1 : int(i) + 1 + block_size]).astype(np.int64)) for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
