import math

import torch
from torch import nn

from einops import rearrange

from model.transformer import PositionalEmbedding


def rope(x, ts=None, inverse=False, maxlen=4096):
    B, T, _, D = x.shape
    if ts is None:
        ts = torch.ones(B, 1, dtype=torch.int32, device=x.device) * torch.arange(T, device=x.device)[None, :]  # [B, T]
    assert ts.shape == (B, T), (ts.shape, (B, T))
    if inverse:
        ts = -ts

    freq_exponents = (2.0 / D) * torch.arange(D // 2, device=x.device)  # [D/2]
    timescale = maxlen ** freq_exponents
    radians = ts[:, :, None] / timescale[None, None, :]  # [B, T, D/2]
    radians = radians[..., None, :].to(x.dtype)  # [B, T, 1, D/2]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = torch.split(x, 2, dim=-1)  # [B, T, H, D/2]
    res = torch.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return res


class Attention(nn.Module):
    def __init__(self, dim, heads=8, kv_heads=0, dropout=0.0, position_embedding='sinusoidal', qknorm='none', bias=True, outscale=1.0):
        super(Attention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.dropout_p = dropout
        self.position_embedding = position_embedding
        self.qknorm = qknorm
        self.bias = bias
        self.outscale = outscale
        self.dropout = nn.Dropout(self.dropout_p)

        kv_heads = self.kv_heads or self.heads
        assert self.heads % kv_heads == 0
        head_ratio = self.heads // kv_heads
        if head_ratio == 1:
            self.qkv = nn.Linear(dim, 3 * self.dim, bias=self.bias)
        else:
            self.q = nn.Linear(dim, self.dim, bias=self.bias)
            self.k = nn.Linear(dim, self.dim // head_ratio, bias=self.bias)
            self.v = nn.Linear(dim, self.dim // head_ratio, bias=self.bias)

        self.normq = nn.Identity()
        self.normk = nn.Identity()
        if self.qknorm == 'layer':
            self.normq = nn.LayerNorm(self.dim // self.heads)
            self.normk = nn.LayerNorm(self.dim // kv_heads)

        self.proj = nn.Linear(self.dim, self.dim, bias=self.bias)

    def forward(self, x, mask=None, ts=None):
        B, T, D = x.shape
        kv_heads = self.kv_heads or self.heads
        assert self.heads % kv_heads == 0
        head_ratio = self.heads // kv_heads
        if head_ratio == 1:
            qkv = self.qkv(x)
            q, k, v = torch.split(qkv, D, dim=-1)
        else:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)

        q = q.reshape(B, T, self.heads, D // self.heads)
        k = k.reshape(B, T, kv_heads, D // kv_heads)
        v = v.reshape(B, T, kv_heads, D // kv_heads)

        q = self.normq(q)
        k = self.normk(k)

        if self.position_embedding == 'rope':
          q = rope(q, ts)
          k = rope(k, ts)

        q = q.reshape(B, T, kv_heads, -1, q.shape[-1])
        logits = torch.einsum('bqhgd,bkhd->bhg q k', q, k)
        logits = logits * (1.0 / math.sqrt(k.shape[-1]))
        if mask is not None:
            Tq, Tk = q.shape[1], k.shape[1]
            assert mask.shape == (B, Tq, Tk), (mask.shape, (B, Tq, Tk))
            mask = mask[:, None, None, ...]
            logits = torch.where(mask, logits, -1e30)
        weights = torch.softmax(logits, dim=-1)
        weights = weights.to(x.dtype)
        weights = self.dropout(weights)
        x = torch.einsum('bhgqk,bkhd->bqhgd', weights, v)
        x = x.reshape(B, T, -1)
        x = self.proj(x)
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dim = cfg.d_model
        self.n_layers = cfg.n_layers
        self.heads = cfg.num_heads
        self.ffup = cfg.get('ffup', 4)
        self.act_name = cfg.activation
        self.norm = 'layer' if cfg.pre_lnorm else 'none'
        self.glu = cfg.get('glu', False)
        self.position_embedding = cfg.get('position_embedding', 'none')
        self.qknorm = 'layer' if cfg.pre_lnorm else 'none'
        self.bias = cfg.get('bias', True)
        self.outscale = cfg.get('outscale', 1.0)
        self.concatenate_over_layers = cfg.deter_type == 'concat_o'
        self.normalize_out = cfg.last_ln
        self.dropout_p = cfg.dropout
        self.drop = nn.Dropout(self.dropout_p)
        self.pos_embs = PositionalEmbedding(self.dim)
        self.layers = nn.ModuleList([self.build_layer() for _ in range(self.n_layers)])
        if self.normalize_out and self.norm == 'layer':
            k = self.n_layers if self.concatenate_over_layers else 1
            self.outnorm = nn.LayerNorm(k * self.dim)

        supported_position_embeddings = {'none', 'rope'}
        assert self.position_embedding in supported_position_embeddings, f'{self.position_embedding} is not in {supported_position_embeddings}'

    def build_layer(self):
        act = {'silu': nn.SiLU(), 'relu': nn.ReLU()}[self.act_name]
        norm1 = nn.Identity()
        norm2 = nn.Identity()
        if self.norm == 'layer':
            norm1 = nn.LayerNorm(self.dim)
            norm2 = nn.LayerNorm(self.dim)

        mha = Attention(self.dim, self.heads, kv_heads=0, dropout=self.dropout_p, position_embedding=self.position_embedding,
                             qknorm=self.qknorm, bias=self.bias, outscale=self.outscale)

        modules = nn.ModuleDict()
        modules['act'] = act
        modules['norm1'] = norm1
        modules['norm2'] = norm2
        modules['mha'] = mha
        if self.glu:
            U = max(self.dim, int((self.dim * self.ffup * 2 / 3) // 32 * 32))
            ff1 = nn.Linear(self.dim, U, bias=self.bias)
            ff2 = nn.Linear(U, U, bias=self.bias)
            ff3 = nn.Linear(U, self.dim, bias=self.bias)
            modules['ff1'] = ff1
            modules['ff2'] = ff2
            modules['ff3'] = ff3
        else:
            ff1 = nn.Linear(self.dim, self.dim * self.ffup, bias=self.bias)
            ff2 = nn.Linear(self.dim * self.ffup, self.dim, bias=self.bias)
            modules['ff1'] = ff1
            modules['ff2'] = ff2

        return modules

    def _generate_square_subsequent_mask(self, B, T, H, W, device):
        N = H * W
        mask = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(0, 1)
        mask = torch.repeat_interleave(mask, N, dim=0)
        mask = torch.repeat_interleave(mask, N, dim=1)
        mask = mask[None, ...]
        mask = mask.expand(B, -1, -1)

        return mask

    def forward(self, x, actions):
        B, T, D, H, W = x.shape
        assert D == self.dim, (D, self.dim)

        mask = self._generate_square_subsequent_mask(B, T, H, W, x.device)  # B, T, T

        # (T, 1, d_model)
        pos_ips = torch.arange(T * H * W, dtype=torch.float).to(x.device)
        pos_embs = self.drop(self.pos_embs(pos_ips))

        if actions is None:
            x = rearrange(x, 'b t d h w -> (t h w) b d')
            x = x + pos_embs

        else:
            x = rearrange(x, 'b t d h w -> (t h w) b d')
            actions = rearrange(actions, 'b t d -> t b d')
            x = self.input_embedding(x) + pos_embs
            action_emb = self.action_embedding(actions)
            action_emb = torch.repeat_interleave(action_emb, H * W, dim=0)
            x += action_emb

        x = x.permute(1, 0, 2)
        out = []
        for layer_modules in self.layers:
            skip = x
            x = layer_modules['norm1'](x)
            x = layer_modules['mha'](x, mask, ts=None)
            x += skip
            skip = x
            x = layer_modules['norm2'](x)
            if self.glu:
                x = layer_modules['ff3'](layer_modules['act'](layer_modules['ff1'](x)) * layer_modules['ff2'](x))
            else:
                x = layer_modules['ff2'](layer_modules['act'](layer_modules['ff1'](x)))

            x += skip
            out.append(x)

        if self.concatenate_over_layers:
            x = torch.stack(out, dim=2)
            x = x.flatten(start_dim=2)

        if self.normalize_out:
            x = self.outnorm(x)

        return x
