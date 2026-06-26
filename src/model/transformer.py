import math
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange, einsum


from src.model.activations import softmax, SiLU


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.zeros(
            size=(out_features, in_features),
            dtype=dtype,
            device=device
            ))
        sigma = math.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma,
        )
    
    def forward(self, x):
        out = einsum(self.weight, x," out in, ... in -> ... out")
        return out

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.device = device
        self.dtype = dtype

        self.emb = nn.Parameter(
            torch.zeros(
                size=(num_embeddings, embedding_dims),
                device=device,
                dtype=dtype
            )
        )
        torch.nn.init.trunc_normal_(
            self.emb,
            mean=0.0,
            std=1.0,
            a=-3,
            b=3,
        )

    def forward(self, token_ids):
        out = self.emb[token_ids, :]
        return out
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device)
        )
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = ((1/self.d_model) * (x ** 2).sum(dim=-1, keepdim=True) + self.eps) ** 0.5
        out = x / rms * self.gain
        return out.to(in_dtype)
        

class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff else int((8/3) * d_model)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype) 
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        y = self.w3(x)
        z = self.w1(x)
        silu = SiLU(z)
        element = y * silu
        out = self.w2(element)
        return out
    

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta # angle for rope
        self.d_k = d_k # dimension of query and key vectors
        self.max_seq_len = max_seq_len # max seq len of input
        self.device = device # device to store buffer on

        k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        denominator = torch.pow(theta, -k / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        rots = torch.outer(positions, denominator)
        self.register_buffer('rotations', rots, persistent=False)


    def forward(self, x, token_positions):
        # extract the rotations from token_positions
        rots = self.rotations[token_positions,:]
        # seperate even and odds
        evens, odds = x[..., 0::2], x[..., 1::2] 
        # apply rotation formula
        evens_new = evens * torch.cos(rots) - odds * torch.sin(rots)
        odds_new = evens * torch.sin(rots) + odds * torch.cos(rots)
        # combine them
        out = torch.stack([evens_new, odds_new], dim=-1)
        # make output dim equal to input dims otherwise dk//2, 2
        out = out.flatten(start_dim=-2)
        return out


def scaled_dot_product_attention(queries, keys, values, mask=None):

    dk = keys.shape[-1]
    scale = dk ** 0.5

    attn = einsum(queries, keys, "... m dk, ... n dk -> ... m n")
    scaled_attn = attn / scale
    if mask is not None:
        scaled_attn[~mask] = float("-inf")
    normalized_scaled_attn = softmax(scaled_attn, -1)

    out = einsum(normalized_scaled_attn, values,"... m n, ... n dk -> ... m dk")
    return out


class MHSA(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.device = device
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.rope = None
        if theta and max_seq_len:
            self.rope = RoPE(theta, self.head_dim, max_seq_len, device)
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.o_proj = Linear(d_model, d_model, device=device)

    def forward(self, x, token_positions=None):
        q = self.q_proj(x)
        q = rearrange(
            q,
            "... s (nh nd) -> ... nh s nd ",
            nh=self.num_heads,
            nd=self.head_dim
        )
        k = self.k_proj(x)
        k = rearrange(
            k,
            "... s (nh nd) -> ... nh s nd ",
            nh=self.num_heads,
            nd=self.head_dim
        )
        v = self.v_proj(x)
        v = rearrange(
            v,
            "... s (nh nd) -> ... nh s nd ",
            nh=self.num_heads,
            nd=self.head_dim
        )
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        dk = self.head_dim
        scale = dk ** 0.5
        attn = einsum(q, k,"... h m dh, ... h n dh -> ... h m n")
        scaled_attn = attn / scale
        L = attn.shape[-1]
        mask = torch.tril(torch.ones(L, L, device=x.device), diagonal=0).to(torch.bool)
        scaled_attn = torch.masked_fill(scaled_attn, ~mask, float("-inf"))
        normalized_scaled_attn = softmax(scaled_attn, -1)

        out = einsum(normalized_scaled_attn, v,"... h m n, ... h n d -> ... h m d")
        out = rearrange(
            out,
            "... nh s hd -> ... s (nh hd)",
            nh=self.num_heads,
            hd = self.head_dim
            )
        out = self.o_proj(out)
        
        return out
    
class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            d_ff,
            theta=None,
            max_seq_len=None,
            device=None,
            dtype=None,
            norm_position: str = "pre",
        ):
        super().__init__()
        assert norm_position in ("pre", "post"), f"norm_position must be 'pre' or 'post', got {norm_position!r}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.norm_position = norm_position
        self.rmsnorm1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.rmsnorm2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.mhsa = MHSA(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
        )
        self.ppff = PointWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
            )

    def forward(self, x, token_positions=None):
        if token_positions is None:
            batch, seq_len, d_model = x.shape
            token_positions = torch.arange(seq_len, device=x.device)
        if self.norm_position == "pre":
            x = x + self.mhsa(self.rmsnorm1(x), token_positions)
            x = x + self.ppff(self.rmsnorm2(x))
        else:  # "post"
            x = self.rmsnorm1(x + self.mhsa(x, token_positions))
            x = self.rmsnorm2(x + self.ppff(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None,
            dtype=None,
            norm_position: str = "pre",
            ):
        super().__init__()
        assert norm_position in ("pre", "post"), f"norm_position must be 'pre' or 'post', got {norm_position!r}"
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = rope_theta
        self.device = device
        self.dtype = dtype
        self.norm_position = norm_position

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dims=d_model,
            device=device,
            dtype=dtype
        )

        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
                norm_position=norm_position,
            ) for _ in range(num_layers)
        )

        # Pre-norm needs a final norm before the LM head; post-norm already normalized at the last block.
        self.rmsnorm = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        ) if norm_position == "pre" else nn.Identity()

        self.linear = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )
   
    def forward(self, x, token_positions=None):
    
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, token_positions)
        
        x = self.linear(self.rmsnorm(x))

        return x


def temperature_scaling(logits: torch.Tensor, temperature: float):
    # logits [batch, vocab_size]
    logits /= temperature
    softmax_l_t = softmax(logits)
    return softmax_l_t

def decoding(
        model: torch.nn.Module, 
        idx: torch.Tensor, 
        max_new_tokens: int,
        context_size: int,
        temperature: int=0,
        eos_token_id: Optional[int]=None
    ):
    """
    Takes in a sample prompt (already converted to token ids). Runs this through the model
    and generates the next tokens. This is done till max token length or EOS sequence is reached.
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # crop the current idx to context_size if it exceeds the supported size
        idx_cropped = idx[:, -context_size:]

        # get the predictions
        with torch.no_grad():
            logits = model(idx_cropped)
        
        # logits [B, n_token, vocab_size]
        logits = logits[:, -1, :] # last timestep
        
        if temperature == 0:
            # Greedy search bypasses softmax entirely
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # scale logits
            scaled_logits = logits / max(temperature, 1e-5)
            
            # convert logits into probabilities
            probs = softmax(scaled_logits, dim=-1)
            
            # sample next token token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

        # append to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

        # if last token is eos - break
        if eos_token_id is not None and idx_next.item() == eos_token_id:
            break
    
    return idx

def generate(
        model: torch.nn.Module,
        checkpoint_path: str,
        tokenizer: callable,
        temperature: int,
        context_size: int=256,
        sample_prompt: str= "Once upon a time",
        device: str='cpu'
):
    # Load model
    
    vals = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(vals["model"])
    model.to(device)
    model.eval()
    
    ids = tokenizer.encode(sample_prompt)
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    eos_id = tokenizer.encode("<|endoftext|>")[0]
    new_idx = decoding(
        model,
        idx,
        max_new_tokens=500,
        eos_token_id=eos_id,
        context_size=context_size,
        temperature=temperature
    )

    output_prompt = tokenizer.decode(new_idx.squeeze(0).tolist())
    return output_prompt


if __name__ == '__main__':
    from experiments.pretraining import get_model_config
    from src.tokenizer.tokenizer_fast import BPE
    save_bpe_path = "/Users/aun/Documents/assignment1-basics/cs336_basics/weights/TinyStoriesV2-GPT4-train-BPE.json"
    checkpoint_path = '/Users/aun/Documents/llms-from-scratch/data/checkpoint_final_step_16517.pt'
    model_config = get_model_config()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = TransformerLM(**model_config, device=device)
    tokenizer = BPE().load_from_file(save_bpe_path)

    output_prompt = generate(
        model,
        checkpoint_path,
        tokenizer,
        temperature=0.05,
        device=device
    )
    print(output_prompt)
