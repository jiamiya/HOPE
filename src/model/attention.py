
import os
# os.environ["SDL_VIDEODRIVER"]="dummy"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
from torch import nn
from einops import rearrange
from tqdm import trange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x: [b, n, f]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [b, h, n, n]

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # v: [b, h, n, d]   attn: [b, h, n, n]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) #  [b, n, h*d] -> [b, n, f]
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(), # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class AttentionNetwork(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, n_features, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim,)
        self.output = nn.Sequential(
            nn.Linear(n_features*dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh(),
        )
        self.view_embed = nn.Parameter(torch.zeros(1, n_features, dim))

    def forward(self, x):
        # x = x + self.view_embed
        x = self.encoder(x)
        x = rearrange(x, 'b n d -> b (n d)')
        return self.output(x)
    
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    feature_dim = 8
    n_feature = 2
    output_dim = 1

    # model = Transformer(
    #     dim = feature_dim,
    #     depth = 6,
    #     heads = 16,
    #     dim_head = 64,
    #     mlp_dim = 256,)
    model = AttentionNetwork(
        dim = feature_dim,
        depth = 6,
        heads = 16,
        dim_head = 64,
        mlp_dim = 256,
        n_features=n_feature,
        hidden_dim=128,
        output_dim=output_dim,
        ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for e in range(100):
        total_loss = 0
        for _ in trange(100):
            data_input = torch.randn(1, n_feature, feature_dim).to(device)
            # data_label = torch.ones((output_dim)) if \
            #     torch.sum(data_input[:,:,:feature_dim//2]) > torch.sum(data_input)/2 else torch.zeros((output_dim))
            data_label = torch.ones((output_dim)).to(device) if \
                torch.sum(data_input[:,:n_feature//2,:]) > torch.sum(data_input)/2 else torch.zeros((output_dim)).to(device)
            data_label = data_label.unsqueeze(0)
            data_output = model(data_input)
            loss = criterion(data_output, data_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('epoch :', e, total_loss/100)

    