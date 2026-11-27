import torch 
import torch.nn as nn

class InputEmbedding(nn.Module):
    """
    Embeds mesh coordinates g and optional physics fields u into  C channels.

    Paper Section 3.2: x^0 = Linear(Concat(g, u))
    """

    def __init__(self, c_g: int, c_u:int, c_hidden:int):
        super().__init__()
        self.proj = nn.Linear(c_g + c_u, c_hidden)

    def forward(self, g :torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        
        if u is not None:
            x = torch.cat([g, u], dim=-1)
        else:
            x = g

        return self.proj(x)

class PhysicsAttention(nn.Module):
    """ PhysicsAttention starting single-head"""

    def __init__(self, c_hidden: int, n_slices: int):
        super().__init__()
        self.n_slices = n_slices
        self.c_hidden = c_hidden
        self.slice_proj = nn.Linear(c_hidden, n_slices)

        self.scale = c_hidden ** -0.5

        self.to_q = nn.Linear(c_hidden ,c_hidden, bias=False)
        self.to_k = nn.Linear(c_hidden ,c_hidden, bias=False)
        self.to_v = nn.Linear(c_hidden ,c_hidden, bias=False)

        self.out_proj = nn.Linear(c_hidden ,c_hidden)

    def encode_slices(self, x: torch.Tensor):

        slice_weigths = torch.softmax(self.slice_proj(x), dim=-1)

        weights_t = slice_weigths.transpose(1, 2)

        slice_tokens = weights_t @ x

        slice_norm = weights_t.sum(dim=-1, keepdim=True)
        slice_tokens = slice_tokens / (slice_norm + 1e-5)

        return slice_tokens, slice_weigths

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        slice_tokens, slice_weights = self.encode_slices(x)

        q = self.to_q(slice_tokens)
        k = self.to_k(slice_tokens)
        v = self.to_v(slice_tokens)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        updated_tokens = attn_weights @ v

        out = slice_weights @ updated_tokens

        return self.out_proj(out)


if __name__ == "__main__":
    B, N, c_g, c_u, c_hidden = 2, 25000, 3, 2, 128

    g = torch.randn(B, N, c_g)
    u = torch.randn(B, N, c_u)

    embed = InputEmbedding(c_g=c_g, c_u=c_u, c_hidden=c_hidden)
    x0 = embed(g, u)

    print("g shape: ", g.shape)
    print("u shape: ", u.shape)
    print("x0 shape: ", x0.shape)

    attn = PhysicsAttention(c_hidden=c_hidden, n_slices=32)
    tokens, weights = attn.encode_slices(x0)

    print("slice_weights shape: ", weights.shape)
    print("slice_tokens shape: ", tokens.shape)
    print("node 0 weight sum: ", weights[0, 0].sum().item())

    x_out = attn(x0)
    print("PhysicsAttention output shape: ", x_out.shape)
