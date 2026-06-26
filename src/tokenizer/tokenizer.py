import json
import regex as re
from collections.abc import Iterable, Iterator

# The net effect: words, numbers, punctuation, and contractions each become separate chunks before BPE sees them, so the merge algorithm can't
# create tokens that span across e.g. a word and a punctuation mark. This is why \p{L} and \p{N} require the regex library rather than stdlib
# re — they're Unicode property classes.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids: list[int], counts: dict[tuple[int, int], int] | None = None) -> dict[tuple[int, int], int]:
    """
    Count adjacent token pair frequencies in ids, accumulating into counts if provided.
    """
    counts = {} if counts is None else counts
    # count adj pair of characters
    for id0, id1 in zip(ids, ids[1:]):
        counts[(id0, id1)] = counts.get((id0, id1), 0) + 1
    return counts

def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    Replace all occurrences of pair in ids with idx.
    """
    i = 0
    newids = []
    # iterate over ids
    while i < len(ids):
        # if match found append newidx
        if i + 1 < len(ids) and pair[0] == ids[i] and pair[1] == ids[i+1]:
            newids.append(idx)
            i += 2
        # append original id
        else:
            newids.append(ids[i])
            i += 1
    return newids


class BPE:
    """Byte-Pair Encoding tokenizer supporting training, encoding, and decoding."""

    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: dict[tuple[int, int], int] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab if vocab else None
        self.merges = merges if merges else None
        self.special_tokens = special_tokens if special_tokens else None
        self.inverse_special_tokens: dict[bytes, int] | None = None

    def load(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]] | dict[tuple[int, int], int],
        special_tokens: list[str] = [],
    ) -> None:
        """
        Load a pre-trained vocab and merge table into the tokenizer.
        """
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = special_tokens
        self.merges = {}
        # merges are in-order when the merge occured.
        
        if isinstance(merges, list):
            for i, pair in enumerate(merges):
                self.merges[self.inverse_vocab[pair[0]], self.inverse_vocab[pair[1]]] = self.inverse_vocab[pair[0] + pair[1]]
        else:
            self.merges = merges

        n = len(self.vocab)
        self.inverse_special_tokens = {}
        # this is to handle special tokens
        # if the tokenizer was trained with special tokens
        # if trained with special tokens use existing id
        # else assign new id 
        if self.special_tokens is not None:
            for i, token in enumerate(self.special_tokens):
                found = False
                token_bytes = token.encode("utf-8")
                for id, bytes in self.vocab.items():
                    if bytes == token_bytes:
                        self.inverse_special_tokens[token_bytes] = id
                        found = True
                        break
                if not found:
                    self.inverse_special_tokens[token_bytes] = n + i

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: list[str] = [],
        verbose: bool = False,
    ) -> None:
        """
        Learn BPE merges from text until vocab_size is reached.
        """
        # combine all special tokens
        if self.special_tokens:
            special_tokens = special_tokens + self.special_tokens

        # number of merges should consider count of special tokens as well
        num_merges = vocab_size - 256 - len(special_tokens)
        # handle special tokens in input text
        # using regex pattern split the text if it has a special token
        special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)
        # keep track of all the special tokens in a lookup table
        # 256 + i (i is from 0 to N) where N is the number of special tokens specified
        special_bank = {special_token: i for i, special_token in enumerate(special_tokens)}
        

        # iterate over chunks of text
        # store them as a list of list of text chunks
        chunks = []
        for special_chunk in special_chunks:
            # if text is a special token leave it as is
            if special_chunk in special_bank:
                chunks += [special_chunk]
            # else use regex pattern to split text on words, numbers, punctuation, and contractions
            else:
                chunks += re.findall(PAT, special_chunk)
        
        # go over these chunks of text 
        ids = []
        for ch in chunks:
            # if not a special character
            # convert it into bytes
            if ch not in special_bank:
                ids.append(list(ch.encode("utf-8")))
            # if it is special character assign correct id from lookup table
            else:
                ids.append([256 + special_bank[ch]])

        merges: dict[tuple[int, int], int] = {}  # merged pair -> id mapping
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} # stores id -> bytes
        # store idx -> bytes
        for sp, i in special_bank.items():
            vocab[256 + i] = sp.encode("utf-8")
        # stores the bytes -> id for special tokens
        self.inverse_special_tokens = {sp.encode("utf-8"): 256 + i for sp, i in special_bank.items()}

        # iterate till we get the required vocab size.
        for n in range(num_merges):
            if verbose:
                print(f"{n} merges")
            # find the most common pair
            stats: dict[tuple[int, int], int] = {}
            for id in ids:
                get_stats(id, stats)
            # get the max count if it is a tie use lexicography to break the tie
            pair = max(stats, key=lambda x: (stats.get(x), vocab[x[0]], vocab[x[1]]))
            # generate new idx
            # remember after the 256 bytes we have special tokens and then new tokens
            idx = 256 + len(special_tokens) + n
            # replace the most common pair with new idx
            ids = [merge(id, pair, idx) for id in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # store the merges, vocabs, tokens, and inverse_vocab for encoding and decoding
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens
        # inverse vocab stores bytes to id pairs
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _encode_chunk(self, byt: bytes) -> list[int]:
        """
        Apply BPE merges to a single pre-tokenized chunk of bytes.
        An example of bytes object is  b"hello"
        """
        # convert bytes objects to ids from vocab
        ids = [self.inverse_vocab[bytes([b])] for b in byt]
        
        # iterate over pairs
        while len(ids) >= 2:
            # get counts of ids
            stats = get_stats(ids)
            # apply the earliest merges FIRST!
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # break if there are no candidates pairs to merge
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            # replace the pair with the idx
            ids = merge(ids, pair, idx)
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """
        Encode text without handling special tokens.
        """
        # split chunk on the regex pattern based words, punctuation, whitespaces, numbers etc
        text_chunks = re.findall(PAT, text)
        enc = []
        # iterate over the chunks
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            enc.extend(chunk_ids)
        return enc

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token ids, preserving special tokens.
        """
        # if special tokens are present
        if self.special_tokens:
            # break the text the same way as we did in the train step
            special_pattern = "(" + "|".join(re.escape(k) for k in sorted(self.special_tokens, key=lambda x: -len(x))) + ")"
            special_chunks = re.split(special_pattern, text)
            special_bank = {special_token: i for i, special_token in enumerate(self.special_tokens)}
        else:
            special_chunks = [text]
            special_bank = {}

        
        encoding = []
        # iterate over chunks
        for chunk in special_chunks:
            # if chunk is a special chunk
            # get correspoining id
            if chunk in special_bank:
                encoding.append(self.inverse_special_tokens[chunk.encode("utf-8")])
            else:
                encoding.extend(self._encode_ordinary(chunk))
        return encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings, yielding token ids one at a time.
        """
        for it in iterable:
            ids = self.encode(it)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token ids back to a string.
        """
        part_bytes = []
        # iterate over ids
        for id in ids:
            # for every id get corresponding bytes
            part_bytes.append(self.vocab[id])
        
        # join the bytes 
        text_bytes = b"".join(part_bytes)
        
        # and decode
        return text_bytes.decode("utf-8", errors='replace')
    
    def save(self, filepath: str) -> None:
        """
        Save the trained tokenizer state to a JSON file.
        """
        if self.vocab is None or self.merges is None:
            raise ValueError("Tokenizer must be trained or loaded before saving.")
            
        # Serialize vocab: convert int keys to strings for JSON compliance
        serializable_vocab = {str(k): v.decode("utf-8", errors="ignore") for k, v in self.vocab.items()}
        
        # Serialize merges: convert tuple keys (int, int) to string "int,int"
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        
        state = {
            "vocab": serializable_vocab,
            "merges": serializable_merges,
            "special_tokens": self.special_tokens or []
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

    @classmethod
    def load_from_file(cls, filepath: str) -> "BPE":
        """
        Create and return a new BPE instance from a saved JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # Reconstruct vocab: convert keys back to int, values back to bytes
        vocab = {int(k): v.encode("utf-8") for k, v in state["vocab"].items()}
        
        # Reconstruct merges: split string key "int,int" back into tuple (int, int)
        merges = {}
        for k, v in state["merges"].items():
            p1, p2 = map(int, k.split(","))
            merges[(p1, p2)] = v
            
        tokenizer = cls()
        tokenizer.load(vocab, merges, state["special_tokens"])
        return tokenizer


if __name__ == "__main__":
    example_text = """
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest <|endoftext|>

    """
    special = ["<|endoftext|>"]

    input_path = "/Users/aun/Documents/assignment1-basics/tests/fixtures/corpus.en"
    with open(input_path, 'r', encoding='utf-8') as file:
        all_text = file.read()
    bpe = BPE()
    bpe.train(all_text, 500, special)
    print(type(bpe.merges))
    print(list(bpe.merges.keys())[:3])

    temp = []
    for p0, p1 in bpe.merges.keys():
        temp.append((bpe.vocab[p0], bpe.vocab[p1]))
    print(temp[:3])
