import tiktoken

from sarthllm.data import tokenizer

_tokenizer = tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return _tokenizer.encode(text)


def decode(token_ids: list[int]) -> str:
    return _tokenizer.decode(token_ids)


if __name__ == "__main__":
    text = "hello world"
    token_ids = encode(text)
    print("text:", text)
    print("ids:", token_ids)
    print("decoded:", decode(token_ids))
  