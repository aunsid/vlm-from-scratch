import json
import regex as re
from typing import Iterable, Iterator

# Assuming PAT and get_stats / merge helpers exist or are defined globally.
# If not, sample minimal implementations are assumed for get_stats/merge.
PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE:
    """Byte-Pair Encoding tokenizer supporting training with pre-token counts, encoding, and decoding."""

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
        merges: dict[tuple[int, int], int],
        special_tokens: list[str] = [],
    ) -> None:
        """Load a pre-trained vocab and merge table into the tokenizer."""
        self.vocab = vocab
        
        # Build inverse vocab mapping bytes -> int
        self.inverse_vocab = {bytes([idx]): idx for idx in range(256)}
        for idx, b in self.vocab.items():
            self.inverse_vocab[b] = idx

        self.special_tokens = special_tokens
        if isinstance(merges, list):
            self.merges = {}
            for pair in merges:
                self.merges[(self.inverse_vocab[pair[0]], self.inverse_vocab[pair[1]])] = self.inverse_vocab[pair[0] + pair[1]]
        else:
            self.merges = merges

        # Handle special tokens mapping
        n = len(self.vocab)
        self.inverse_special_tokens = {}
        if self.special_tokens is not None:
            for i, token in enumerate(self.special_tokens):
                found = False
                token_bytes = token.encode("utf-8")
                if token_bytes in self.inverse_vocab:
                    self.inverse_special_tokens[token_bytes] = self.inverse_vocab[token_bytes]
                    found = True
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
        Learn BPE merges from text until vocab_size is reached using pre-token frequency counts.
        """
        if self.special_tokens:
            special_tokens = special_tokens + self.special_tokens

        num_merges = vocab_size - 256 - len(special_tokens)
        
        # Split text into special and regular chunks
        special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)
        special_bank = {special_token: i for i, special_token in enumerate(special_tokens)}
        
        chunks = []
        for special_chunk in special_chunks:
            if not special_chunk:
                continue
            if special_chunk in special_bank:
                chunks.append(special_chunk)
            else:
                chunks.extend(re.findall(PAT, special_chunk))
        
        # --- PRE-TOKEN COUNT LOGIC ---
        # Instead of a global list of IDs, we map unique ID sequences to their frequency counts.
        word_freqs: dict[tuple[int, ...], int] = {}
        for ch in chunks:
            if ch not in special_bank:
                ch_ids = tuple(ch.encode("utf-8"))
            else:
                ch_ids = (256 + special_bank[ch],)
            word_freqs[ch_ids] = word_freqs.get(ch_ids, 0) + 1

        merges: dict[tuple[int, int], int] = {}  
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} 
        
        for sp, i in special_bank.items():
            vocab[256 + i] = sp.encode("utf-8")
        self.inverse_special_tokens = {sp.encode("utf-8"): 256 + i for sp, i in special_bank.items()}

        # Core training loop using frequencies
        for n in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            
            # Count pairs weighted by pre-token occurrences
            for word_ids, freq in word_freqs.items():
                for pair in zip(word_ids, word_ids[1:]):
                    stats[pair] = stats.get(pair, 0) + freq
            
            if not stats:
                if verbose:
                    print("No more merge candidates available.")
                break

            # Find the most common pair (tie-breaking with lexicography)
            pair = max(stats, key=lambda x: (stats.get(x), vocab[x[0]], vocab[x[1]]))
            idx = 256 + len(special_tokens) + n
            
            if verbose:
                print(f"Merge {n}: {pair} -> {idx} (count: {stats[pair]})")

            # Update our frequency dictionary with the newly merged token sequence
            new_word_freqs = {}
            for word_ids, freq in word_freqs.items():
                new_ids = list(word_ids)
                i = 0
                while i < len(new_ids) - 1:
                    if new_ids[i] == pair[0] and new_ids[i+1] == pair[1]:
                        new_ids[i:i+2] = [idx]
                    else:
                        i += 1
                new_word_freqs[tuple(new_ids)] = freq
                
            word_freqs = new_word_freqs
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.inverse_vocab = {bytes([idx]): idx for idx in range(256)}
        for idx, b in self.vocab.items():
            self.inverse_vocab[b] = idx

    def _encode_chunk(self, byt: bytes) -> list[int]:
        ids = [self.inverse_vocab[bytes([b])] for b in byt]
        while len(ids) >= 2:
            # Quick stat gathering for the current sequence
            stats = {}
            for pair in zip(ids, ids[1:]):
                stats[pair] = stats.get(pair, 0) + 1
                
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        text_chunks = re.findall(PAT, text)
        enc = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            enc.extend(chunk_ids)
        return enc
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            ids = self.encode(it)
            for id in ids:
                yield id

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in sorted(self.special_tokens, key=lambda x: -len(x))) + ")"
            special_chunks = re.split(special_pattern, text)
            special_bank = {special_token: i for i, special_token in enumerate(self.special_tokens)}
        else:
            special_chunks = [text]
            special_bank = {}

        encoding = []
        for chunk in special_chunks:
            if not chunk:
                continue
            if chunk in special_bank:
                encoding.append(self.inverse_special_tokens[chunk.encode("utf-8")])
            else:
                encoding.extend(self._encode_ordinary(chunk))
        return encoding
    
    def save(self, filepath: str) -> None:
        """Save the trained tokenizer state to a JSON file safely using latin-1."""
        if self.vocab is None or self.merges is None:
            raise ValueError("Tokenizer must be trained or loaded before saving.")
            
        # Convert bytes values to safe latin-1 strings for JSON serialization
        serializable_vocab = {str(k): v.decode("latin-1") for k, v in self.vocab.items()}
        
        # Convert tuple keys (int, int) to a string "int,int"
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
        """Create and return a new BPE instance from a saved JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # Reconstruct vocab: string keys -> int, string values -> bytes
        vocab = {int(k): v.encode("latin-1") for k, v in state["vocab"].items()}
        
        # Reconstruct merges dict: string key "int,int" -> tuple (int, int)
        merges = {}
        for k, v in state["merges"].items():
            p1, p2 = map(int, k.split(","))
            merges[(p1, p2)] = v
            
        tokenizer = cls()
        tokenizer.load(vocab, merges, state["special_tokens"])
        return tokenizer

    def decode(self, ids: list[int]) -> str:
        part_bytes = []
        for id in ids:
            part_bytes.append(self.vocab[id])
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors='replace')