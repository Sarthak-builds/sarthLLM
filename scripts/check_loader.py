import numpy as np

from sarthllm.data.dataset import GPTDataset, create_dataloader, train_test_split
from sarthllm.data.shards import TokenShards
from sarthllm.data.tokenizer import decode

D = "artifacts/data/fineweb_edu"
shards = TokenShards(D, max_tokens=500_000_000)
print("tokens:", len(shards), "| shards used:", len(shards.paths))

# a read that spans two shards must equal the two files stitched together
b = shards.sizes[0]
a0 = np.memmap(shards.paths[0], dtype=np.uint16, mode="r")
a1 = np.memmap(shards.paths[1], dtype=np.uint16, mode="r")
assert (shards.read(b - 5, b + 5) == np.concatenate([a0[-5:], a1[:5]])).all()

dataset = GPTDataset(shards, max_length=1024)
train_ds, test_ds = train_test_split(dataset, test_ratio=0.01)
assert set(train_ds.indices).isdisjoint(test_ds.indices)
train_loader = create_dataloader(train_ds, batch_size=8, shuffle=True)
test_loader = create_dataloader(test_ds, batch_size=8, shuffle=False)
print("train windows:", len(train_ds), "| batches:", len(train_loader))
print("test windows: ", len(test_ds), "| batches:", len(test_loader))

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.dtype)
assert (x[:, 1:] == y[:, :-1]).all() 
print(decode(x[0, :60].tolist()))
tiny = TokenShards(D, max_tokens=1000)
print(next(iter(create_dataloader(GPTDataset(tiny, 4, stride=1), 1, shuffle=False)))[0])
print(next(iter(create_dataloader(GPTDataset(tiny, 4, stride=4), 1, shuffle=False)))[0])