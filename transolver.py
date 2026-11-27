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

if __name__ == "__main__":
    B, N, c_g, c_u, c_hidden = 2, 500, 3, 2, 128

    g = torch.randn(B, N, c_g)
    u = torch.randn(B, N, c_u)

    embed = InputEmbedding(c_g=c_g, c_u=c_u, c_hidden=c_hidden)
    x0 = embed(g, u)

    print("g shape: ", g.shape)
    print("u shape: ", u.shape)
    print("x0 shape: ", x0.shape)
