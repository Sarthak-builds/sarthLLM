import torch
from torch.utils.data import DataLoader, Dataset, random_split


class Sarthllm_Dataset(Dataset):

    def __init__(self, tokens, max_length, stride=None):
        self.tokens = tokens
        self.max_length = max_length
        self.stride = stride or max_length

    def __len__(self):
        return (len(self.tokens) - self.max_length - 1) // self.stride + 1

    def __getitem__(self, idx):
        i = idx * self.stride
        token_ids = self.tokens.read(i, i + self.max_length + 1)
        input_chunk = torch.from_numpy(token_ids[:-1])
        target_chunk = torch.from_numpy(token_ids[1:]) 
        return input_chunk, target_chunk


def train_test_split(dataset, test_ratio=0.01, seed=2104):
    n_test = int(len(dataset) * test_ratio)
    generator = torch.Generator().manual_seed(seed) 
    return random_split(dataset, [len(dataset) - n_test, n_test], generator=generator)


def create_dataloader(dataset, batch_size=8, shuffle=True, drop_last=True,
                      num_workers=0, seed=2104):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),  
    )