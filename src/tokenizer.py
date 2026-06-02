import tiktoken


class GPT2Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_token = self.enc.eot_token
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        tokens = self.enc.encode_ordinary(text)
        if add_eot:
            tokens.append(self.eot_token)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

