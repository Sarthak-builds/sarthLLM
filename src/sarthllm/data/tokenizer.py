import tiktoken

_tokenizer = tiktoken.get_encoding("gpt2") 
EOT = "<|endoftext|>"
EOT_ID = _tokenizer.eot_token  # 50256

def encode(text: str) -> list[int]:
    return _tokenizer.encode(text)

def decode(token_ids: list[int]) -> str:
    return _tokenizer.decode(token_ids)

def encode_doc(text: str) -> list[int]:
    return _tokenizer.encode(text, disallowed_special=()) + [EOT_ID]

def encode_docs(texts: list[str], num_threads: int = 4) -> list[list[int]]:
    batch = _tokenizer.encode_batch(texts, num_threads=num_threads, disallowed_special=())
    return [ids + [EOT_ID] for ids in batch]