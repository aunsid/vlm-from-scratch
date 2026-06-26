import numpy as np
import torch


def get_batch(
        dataset: np.array,
        batch_size: int = 32,
        context_length: int = 256,
        device: str = 'cpu',
        overfit: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reads batches of data from memory map and returns inputs and target tensors.
    """
    length = dataset.shape[0]
    max_idx = length - context_length - 1

    if overfit:
        g = torch.Generator()
        g.manual_seed(42) 
        ix = torch.randint(0, max_idx + 1, (batch_size,), generator=g)
    else:
        ix = torch.randint(0, max_idx+1, (batch_size, ))

    # Extract the input and target chunks based on the sampled starting positions
    x_list = [torch.from_numpy(dataset[i : i + context_length]) for i in ix]
    y_list = [torch.from_numpy(dataset[i + 1 : i + context_length + 1]) for i in ix]
    
    # Stack the list of 1D tensors into a 2D tensor of shape (batch_size, context_length)
    x = torch.stack(x_list).long().to(device)
    y = torch.stack(y_list).long().to(device)

    return x, y