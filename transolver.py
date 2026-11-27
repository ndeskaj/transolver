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
    """ PhysicsAttention starting multi-head version"""

    def __init__(self, c_hidden: int, n_slices: int, n_heads: int = 8):
        super().__init__()
        self.n_slices = n_slices
        self.c_hidden = c_hidden
        self.n_heads = n_heads
        self.head_dim = c_hidden // n_heads
        self.slice_proj = nn.Linear(self.head_dim, n_slices)

        self.in_proj = nn.Linear(c_hidden, c_hidden)

        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(self.head_dim ,self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim ,self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim ,self.head_dim, bias=False)

        self.out_proj = nn.Linear(c_hidden ,c_hidden)

    def encode_slices(self, x: torch.Tensor):

        B, N, C = x.shape

        x_heads = self.in_proj(x).reshape(B, N, self.n_heads, self.head_dim)
        x_heads = x_heads.transpose(1, 2)

        slice_weigths = torch.softmax(self.slice_proj(x_heads), dim=-1)

        weights_t = slice_weigths.transpose(-2, -1)

        slice_tokens = weights_t @ x_heads

        slice_norm = weights_t.sum(dim=-1, keepdim=True)
        slice_tokens = slice_tokens / (slice_norm + 1e-5)

        return slice_tokens, slice_weigths

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, N, C = x.shape

        slice_tokens, slice_weights = self.encode_slices(x)

        q = self.to_q(slice_tokens)
        k = self.to_k(slice_tokens)
        v = self.to_v(slice_tokens)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        updated_tokens = attn_weights @ v

        out = slice_weights @ updated_tokens
        out = out.transpose(1, 2).reshape(B, N, C)

        return self.out_proj(out)

class TransolverBlock(nn.Module):
    """ One Pre-LN Transolver layer"""

    def __init__(
            self,
            c_hidden: int,
            n_slices: int,
            n_heads: int=8,
            mlp_ratio: int=2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(c_hidden)
        self.attn = PhysicsAttention(c_hidden, n_slices, n_heads)
        self.norm2 = nn.LayerNorm(c_hidden)
        self.ffn = nn.Sequential(
            nn.Linear(c_hidden, c_hidden * mlp_ratio),
            nn.GELU(),
            nn.Linear(c_hidden * mlp_ratio, c_hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Transolver(nn.Module):
    """Full transolver model for PDEs"""

    def __init__(
            self,
            c_g: int,
            c_u: int, 
            c_out: int,
            c_hidden: int=128,
            n_layers: int=8,
            n_heads: int=8,
            n_slices: int=32,
            mlp_ratio: int=2,
    ):
        super().__init__()
        self.embed = InputEmbedding(c_g, c_u, c_hidden)
        self.blocks = nn.ModuleList(
            [TransolverBlock(c_hidden, n_slices, n_heads, mlp_ratio) 
                for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(c_hidden)
        self.head = nn.Linear(c_hidden, c_out)

    def forward(self, g: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(g, u)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head(x)

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

    print("panel 0, head 0, node 0 weight sum: ", weights[0, 0, 0].sum().item())

    # Full Transolver predicting [dx, dy, dz, final_thickness] at every FEM node
    model = Transolver(
        c_g=3, c_u=2, c_out=4,
        c_hidden=128, n_layers=8, n_heads=8, n_slices=32,
    )
    n_params = sum(p.numel() for p in model.parameters())
    pred = model(g, u)

    print("Total parameters  :", f"{n_params:,}")
    print("Forming pred shape:", pred.shape)