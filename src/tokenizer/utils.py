from typing import Iterator, Callable

import numpy as np

from src.tokenizer.tokenizer_fast import BPE
from dataloader.dataloader import get_batch

def training_bpe(
        corpus_path: str,
        vocab_size: int,
        special_tokens: list = [],
        verbose: bool=False,
        save_tokenizer_params_path: str = "../weights/tokenizer.json"
    ):
    """
    Trains BPE tokenizer on the training corpus.
    Saves the config/data in the specified path.
    """
    with open(corpus_path, 'r', encoding='utf-8') as file:
        all_text = file.read()
    tokenizer = BPE()
    tokenizer.train(
        text=all_text,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        verbose=verbose
    )
    tokenizer.save(filepath=save_tokenizer_params_path)
    return tokenizer



def stream_corpus_by_chunks(
        file_path: str,
        chunk_size: int = 65536
    )-> Iterator[str]:
    """
    Streams a large text file in  blocks to avoid RAM overhead.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def tokenize_corpus(
        corpus_path: str,
        tokenizer_param_path: str,
        save_tokens_path: str,
        batch_size:int = 500_000
):
    """
    Tokenizes a large text corpus line/chunk-by-chunk and streams it directly into a binary file.
    """
    tokenizer = BPE.load_from_file(tokenizer_param_path)
    vocab_size = len(tokenizer.vocab)
    
    # choose minimal datatype based on vocabulary size to save disk space
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32

    print(f"Using data type: {dtype.__name__} ({np.dtype(dtype).itemsize} bytes per token)")

    text_stream = stream_corpus_by_chunks(corpus_path)
    token_stream = tokenizer.encode_iterable(text_stream)

    print("Starting offline tokenization pass...")
    
    buffer = []
    total_tokens = 0

    # Open file in append binary mode
    with open(save_tokens_path, "wb") as f_out:
        for token_id in token_stream:
            buffer.append(token_id)
            
            # Flush batch to disk periodically
            if len(buffer) >= batch_size:
                arr = np.array(buffer, dtype=dtype)
                f_out.write(arr.tobytes())
                total_tokens += len(buffer)
                buffer = []
                print(f"Processed {total_tokens:,} tokens...")

        # Flush any remaining tokens left in the buffer
        if buffer:
            arr = np.array(buffer, dtype=dtype)
            f_out.write(arr.tobytes())
            total_tokens += len(buffer)

    print(f"\nDone! Saved {total_tokens:,} tokens to '{save_tokens_path}'.")


if __name__ == "__main__":
    import torch
    # Paths
    corpus_path = "/Users/aun/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    save_bpe_path = "/Users/aun/Documents/assignment1-basics/cs336_basics/weights/TinyStoriesV2-GPT4-train-BPE.json"
    tokenized_data_path = "/Users/aun/Documents/assignment1-basics/cs336_basics/weights/TinyStoriesV2-GPT4-train-BPE-tokenized.bin"
    
    # Train Tokenizer
    special = ["<|endoftext|>"]
    tokenizer = training_bpe(
        corpus_path=corpus_path,
        vocab_size=10000,
        special_tokens=special,
        verbose=True,
        save_tokenizer_params_path=save_bpe_path,
    )
    
    # tokenize corpus and save
    tokenize_corpus(corpus_path, save_bpe_path, tokenized_data_path)
    vocab_size = len(tokenizer.vocab)
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    tokenized_data = np.memmap(tokenized_data_path, dtype=np.uint16, mode="r")
    tokens=541271275
    # check if batches are same for overfitting
    xb1, yb1 = get_batch(
        dataset=tokenized_data, batch_size=32,  context_length=256, device='cpu', overfit=True
    )

    xb2, yb2 = get_batch(
        dataset=tokenized_data, batch_size=32,  context_length=256, device='cpu', overfit=True
    )
    # Check if the inputs match
    inputs_match = torch.equal(xb1, xb2)

    # Check if the targets match
    targets_match = torch.equal(yb1, yb2)

    print(f"Are input batches identical? {inputs_match}")
    print(f"Are target batches identical? {targets_match}")

    # Or check both at once:
    if torch.equal(xb1, xb2) and torch.equal(yb1, yb2):
        print("Success! The batches are perfectly identical.")
    else:
        print("Failure: The batches are different.")

    