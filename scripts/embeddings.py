import torch

from sarthllm.data.dataset import Sarthllm_Dataset, create_dataloader
from sarthllm.data.shards import TokenShards
from sarthllm.model.config import SARTHLLM_CONFIG

torch.manual_seed(2104)

# for check purpose we take 50M only and with batch size of 8 only.....
shards = TokenShards("artifacts/data/fineweb_edu", max_tokens=50_000_000)
dataset = Sarthllm_Dataset(shards, max_length=SARTHLLM_CONFIG["context_length"])
loader = create_dataloader(dataset, batch_size=8, shuffle=False)
inputs, targets = next(iter(loader))
print("token ids:", inputs[0].shape)

token_embedding_layer = torch.nn.Embedding(SARTHLLM_CONFIG["vocab_size"], SARTHLLM_CONFIG["emb_dim"])
token_embeddings = token_embedding_layer(inputs)
print("token embeddings:", token_embeddings.shape)

pos_embedding_layer = torch.nn.Embedding(SARTHLLM_CONFIG["vocab_size"], SARTHLLM_CONFIG["emb_dim"])
pos_embeddings = pos_embedding_layer(torch.arange(SARTHLLM_CONFIG["context_length"]))
print("pos embeddings:", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings