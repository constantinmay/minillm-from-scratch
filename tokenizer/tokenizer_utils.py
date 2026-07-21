"""MiniLLM Tokenizer wrapper around HuggingFace tokenizers BPE."""

from typing import List, Optional

from tokenizers import Tokenizer


class MiniLLMTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        self.tokenizer.enable_padding(pad_id=self.pad_id(), pad_token=self.PAD_TOKEN)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        encodings = self.tokenizer.encode_batch(texts)
        return [e.ids for e in encodings]

    def decode_batch(self, ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(self.PAD_TOKEN)

    def unk_id(self) -> int:
        return self.tokenizer.token_to_id(self.UNK_TOKEN)

    def bos_id(self) -> int:
        return self.tokenizer.token_to_id(self.BOS_TOKEN)

    def eos_id(self) -> int:
        return self.tokenizer.token_to_id(self.EOS_TOKEN)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, idx: int) -> str:
        return self.tokenizer.id_to_token(idx)
