from pathlib import Path

import numpy as np
import torch


class SupervisedTokenDataset:
    def __init__(self, tokens_path: str | Path, labels_path: str | Path):
        self.tokens_path = Path(tokens_path)
        self.labels_path = Path(labels_path)
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Token file not found: {self.tokens_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Label file not found: {self.labels_path}")

        self.tokens = np.memmap(self.tokens_path, dtype=np.uint16, mode="r")
        self.labels = np.memmap(self.labels_path, dtype=np.int32, mode="r")
        if self.tokens.shape[0] != self.labels.shape[0]:
            raise ValueError("tokens and labels must have the same length")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device | str,
        max_tries: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.tokens) <= block_size + 1:
            raise ValueError(f"{self.tokens_path} must contain more than block_size + 1 tokens")

        xs = []
        ys = []
        max_start = len(self.tokens) - block_size - 1
        for _ in range(batch_size):
            for _ in range(max_tries):
                start = int(torch.randint(max_start, (1,)).item())
                y_np = self.labels[start + 1 : start + 1 + block_size].astype(np.int64)
                if np.any(y_np != -100):
                    break
            else:
                start = int(torch.randint(max_start, (1,)).item())
                y_np = self.labels[start + 1 : start + 1 + block_size].astype(np.int64)

            x_np = self.tokens[start : start + block_size].astype(np.int64)
            xs.append(torch.from_numpy(x_np))
            ys.append(torch.from_numpy(y_np))

        return torch.stack(xs).to(device, non_blocking=True), torch.stack(ys).to(device, non_blocking=True)
