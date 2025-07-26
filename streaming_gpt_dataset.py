# gpt_dataset.py
import torch
import numpy as np
from torch.utils.data import IterableDataset

class StreamingGPTDataset(IterableDataset):
    def __init__(self, file_path, block_size=1024, stride=128, start_offset=0):
        super().__init__()
        self.file_path = file_path
        self.block_size = block_size
        self.stride = stride
        self.start_offset = start_offset

    def __iter__(self):
        tokens = np.memmap(self.file_path, dtype=np.int32, mode='r')
        start = self.start_offset
        end = len(tokens) - self.block_size

        for i in range(start, end, self.stride):
            chunk = tokens[i : i + self.block_size]
            yield torch.tensor(chunk, dtype=torch.long)
