import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class MLP(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, bias: bool = True):
        """
        MLP
        d_in (int): input size
        d_h (int): hidden size
        d_out (int): output size

        two-layer mlp with GeLU activation
        input size: (..., d_in)
        output size: (..., d_out)
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        self.l1 = nn.Linear(d_in, d_h)
        self.act = nn.GELU()
        self.l2 = nn.Linear(d_h, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.act(self.l1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        """ "
        MultiHeadAttention
        d_model (int): model dimension
        n_heads (int): number of heads to split d_model across

        does non-causal multi-head attention on input sequence
        input size: (B, T, C)
        output size: (B, T, C)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.rope = RoPE(d=d_model // n_heads)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, C = x.size()
        Q = norm(self.rope(self.wq(x).view(B, T, self.n_heads, -1)).transpose(1, 2))
        K = norm(self.rope(self.wk(x).view(B, T, self.n_heads, -1)).transpose(1, 2))
        V = self.wv(x).view(B, T, self.n_heads, -1).transpose(1, 2)

        attn_scores = (
            F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        return self.wo(attn_scores.view(B, T, C))


class DoubleStreamAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        CrossAttention
        d_model (int): model dimension
        n_heads (int): number of heads to split d_model across

        input size: (B, T, C), (B, N, C)
        output size: (B, T, C), (B, N, C)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0

        self.rope = RoPE(d=d_model // n_heads)
        
        self.wq_text = nn.Linear(d_model, d_model)
        self.wk_text = nn.Linear(d_model, d_model)
        self.wv_text = nn.Linear(d_model, d_model)

        self.wq_image = nn.Linear(d_model, d_model)
        self.wk_image = nn.Linear(d_model, d_model)
        self.wv_image = nn.Linear(d_model, d_model)

        self.wo_text = nn.Linear(d_model, d_model)
        self.wo_image = nn.Linear(d_model, d_model)

    def forward(
        self,
        image_x: torch.Tensor,
        text_x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = image_x.size()

        Q_image = rearrange(self.wq_image(image_x), "b t (h d) -> b t h d", h=self.n_heads)
        K_image = rearrange(self.wk_image(image_x), "b t (h d) -> b t h d", h=self.n_heads)
        V_image = rearrange(self.wv_image(image_x), "b t (h d) -> b t h d", h=self.n_heads)

        Q_text = rearrange(self.wq_text(text_x), "b t (h d) -> b t h d", h=self.n_heads)
        K_text = rearrange(self.wk_text(text_x), "b t (h d) -> b t h d", h=self.n_heads)
        V_text = rearrange(self.wv_text(text_x), "b t (h d) -> b t h d", h=self.n_heads)

        

        Q = self.rope(norm(torch.cat((Q_image, Q_text), dim=1))).transpose(1, 2)
        K = self.rope(norm(torch.cat((K_image, K_text), dim=1))).transpose(1, 2)
        V = norm(torch.cat((V_image, V_text), dim=1)).transpose(1, 2)

        attn_outs = rearrange(F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask), "b h t d -> b t (h d)")

        
        attn_outs_image = attn_outs[:, :N, :]
        attn_outs_text = attn_outs[:, N:, :]

        return self.wo_image(attn_outs_image), self.wo_text(attn_outs_text)


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int):
        """
        SwiGLU
        d_in (int): input size
        d_h (int): hidden size
        d_out (int): output size

        two-layer gated MLP with SwiGLU activation
        input size: (..., d_in)
        output size: (..., d_out)
        """
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        self.W = nn.Linear(d_in, d_h)
        self.V = nn.Linear(d_in, d_h)

        self.l2 = nn.Linear(d_h, d_out)

    def swish_1(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.swish_1(self.W(x)) * self.V(x))


class ConvPatchify(nn.Module):
    def __init__(self, d_model: int, patch_size: int, n_channels: int = 4):
        """
        ConvPatchify
        d_model (int): model dimension
        patch_size (int): patch_size to split into
        n_channels (int): image channels

        input size: (B, C, X, Y)
        output size: (B, T, D)
        """
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.conv = nn.Conv2d(
            n_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), self.d_model, -1).transpose(1, 2)


class Depatchify(nn.Module):
    def __init__(
        self, d_model: int, patch_size: int, w: int, h: int, n_channels: int = 4
    ):
        """
        Depatchify
        d_model (int): model dimension
        patch_size (int): patch_size split into
        w (int): image width
        h (int): image height
        n_channels (int): image channels

        input size: (B, T, C)
        output size: (B, c, H, W)
        """
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.w = w
        self.h = h

        self.proj = nn.Linear(d_model, n_channels * patch_size**2, bias=False)

    def forward(self, x: torch.Tensor, N: int) -> torch.Tensor:
        B, T, C = x.size()
        # we project C -> c * p^2
        # view as B, N, c, p, p
        # transpose to B, c, N, p, p
        # reshape to B, c, H, W
        return rearrange(
            self.proj(x)[:, :N, :],
            "b (npy npx) (c psy psx) -> b c (npy psy) (npx psx)",
            psx=self.patch_size,
            psy=self.patch_size,
            npx=self.w // self.patch_size,
            npy=self.h // self.patch_size,
        )


class SinusoidalEmbedding(nn.Module):
    def __init__(self, n_embeddings: int, d_model: int):
        """
        SinusoidalEmbedding
        n_embeddings (int): number of positions to store
        d_model (int): dimension of each embedding

        input size: (B, T)
        output size: (B, T, C)
        """
        super().__init__()
        self.n_embeddings = n_embeddings
        self.d_model = d_model

        emb = torch.empty(n_embeddings, d_model)
        for pos in range(n_embeddings):
            for i in range(d_model):
                emb[pos, i] = (
                    math.sin(pos / 10000 ** (2 * i / d_model))
                    if i % 2 == 0
                    else math.cos(pos / 10000 ** (2 * i / d_model))
                )
        self.register_buffer("emb", emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb[x]


class RoPE(nn.Module):
    def __init__(self, d: int, base: float = 10000):
        """
        RoPE
        d (int): model dimension
        cache_positions (int): number of precomputed sines/cosines to store
        base (float): base used for calculating thetas

        input size: (B, T, n, d)
        output size: (B, T, n, d)
        """

        super().__init__()
        self.d = d
        self.register_buffer(
            "angles",
            torch.repeat_interleave(base ** (-2 * torch.arange(d // 2) / d), 2, dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, n, d = x.size()
        positions = torch.arange(T, device=x.device)

        position_angles = torch.einsum("t, d -> td", positions, self.angles)

        x_shuffled = torch.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), dim=-1).view(
            B, T, n, d
        )
        return x * torch.cos(position_angles).unsqueeze(0).unsqueeze(
            2
        ) + x_shuffled * torch.sin(position_angles).unsqueeze(0).unsqueeze(2)


class MultiStreamDiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        MultiStreamDiTBlock
        d_model (int): model dimension
        n_heads (int): number of heads to split d_model across

        does cross-attention on image tokens (Q) and text (KV)

        input size: (B, N, C), (B, T, C), (B, 14)
        output size: (B, N, C), (B, T, C)
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.attention = DoubleStreamAttention(d_model=d_model, n_heads=n_heads)

        self.mlp_image = SwiGLU(d_in=d_model, d_h=d_model * 4, d_out=d_model)
        self.mlp_text = SwiGLU(d_in=d_model, d_h=d_model * 4, d_out=d_model)

        
        self.adaln_mlp_image = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        self.adaln_mlp_text = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))

        nn.init.zeros_(self.adaln_mlp_image[1].weight)
        nn.init.zeros_(self.adaln_mlp_image[1].bias)
        nn.init.zeros_(self.adaln_mlp_text[1].weight)
        nn.init.zeros_(self.adaln_mlp_text[1].bias)

    def forward(
        self,
        image_x: torch.Tensor,
        text_x: torch.Tensor,
        condition_input: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = text_x.size()
        N = image_x.size(1)
        condition_image = self.adaln_mlp_image(condition_input).view(B, 6, C)

        condition_text = self.adaln_mlp_text(condition_input).view(B, 6, C)
        
        image_shift_attn, image_scale_pre_attn, image_scale_attn, image_shift_mlp, image_scale_pre_mlp, image_scale_mlp = condition_image.chunk(6, dim=1)
        text_shift_attn, text_scale_pre_attn, text_scale_attn, text_shift_mlp, text_scale_pre_mlp, text_scale_mlp = condition_text.chunk(6, dim=1)

        attn_o_image, attn_o_text = self.attention(norm(image_x) * (1 + image_scale_pre_attn) + image_shift_attn, norm(text_x) * (1 + text_scale_pre_attn) + text_shift_attn, attn_mask=attn_mask)

        image_x = image_x + attn_o_image * image_scale_attn
        image_x = image_x + self.mlp_image(norm(image_x) * (1 + image_scale_pre_mlp) + image_shift_mlp) * image_scale_mlp

        text_x = text_x + attn_o_text * text_scale_attn
        text_x = text_x + self.mlp_text(norm(text_x) * (1 + text_scale_pre_mlp) + text_shift_mlp) * text_scale_mlp

        return image_x, text_x

class SingleStreamDitBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        SingleStreamDitBlock
        d_model (int): model dimension
        n_heads (int): heads to split d_model across

        does regular non-causal attention on concatenated img/text tokens

        input size: (B, T, C)
        output size: (B, T, C)
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        self.mlp = SwiGLU(d_model, d_model, d_model)

        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))

        nn.init.zeros_(self.adaln_mlp[1].weight)
        nn.init.zeros_(self.adaln_mlp[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition_input: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        condition = self.adaln_mlp(condition_input).view(B, 6, C)
        shift_attn, scale_pre_attn, scale_attn, shift_mlp, scale_pre_mlp, scale_mlp = condition.chunk(6, dim=1)
        
        x = x + self.attention(norm(x) * (1 + scale_pre_attn) + shift_attn, attn_mask=attn_mask) * scale_attn
        x = x + self.mlp(norm(x) * (1 + scale_pre_mlp) + shift_mlp) * scale_mlp

        return x


class DiT(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers_multi_stream: int,
        n_layers_single_stream: int,
        patch_size: int,
        w: int,
        h: int,
        clip_encoder_hidden_size: int,
        t5_encoder_hidden_size: int,
        n_timesteps: int = 1000,
        n_channels: int = 4,
    ):
        """
        DiT
        encoder_model (T5 encoder): the text embedding model to use
        d_model (int): model dimension
        n_heads (int): heads to split d_model across
        n_layers_multi_stream (int): number of multi-stream cross-attention layers
        n_layers_single_stream (int): number of regular attention layers
        patch_size (int): patch size for patchify
        w (int): image width
        h (int): image height
        n_timesteps (int): number of timestep embeddings to store
        n_channels (int): number of image channels, usually 3 or 4
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers_multi_stream = n_layers_multi_stream
        self.n_layers_single_stream = n_layers_single_stream
        self.patch_size = patch_size
        self.w = w
        self.h = h
        self.clip_encoder_hidden_size = clip_encoder_hidden_size
        self.t5_encoder_hidden_size = t5_encoder_hidden_size
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels

        self.null_cond_vector = nn.Parameter(torch.zeros(clip_encoder_hidden_size))

        self.embedding_proj = nn.Linear(
            t5_encoder_hidden_size, d_model, bias=False
        )

        self.pool_proj = nn.Linear(clip_encoder_hidden_size, d_model, bias=False)

        self.patchify = ConvPatchify(
            d_model=d_model, patch_size=patch_size, n_channels=n_channels
        )

        self.time_embedding = SinusoidalEmbedding(
            n_embeddings=n_timesteps, d_model=d_model
        )

        self.multi_stream_layers = nn.ModuleList(
            [
                MultiStreamDiTBlock(d_model=d_model, n_heads=n_heads)
                for i in range(n_layers_multi_stream)
            ]
        )

        self.single_stream_layers = nn.ModuleList(
            [
                SingleStreamDitBlock(d_model=d_model, n_heads=n_heads)
                for i in range(n_layers_single_stream)
            ]
        )

        self.final_ln = nn.LayerNorm(d_model)

        self.depatchify = Depatchify(
            d_model=d_model, patch_size=patch_size, w=w, h=h, n_channels=n_channels
        )
        nn.init.xavier_normal_(
            self.depatchify.proj.weight
        )  # tried zero-init, did worse

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        cond_pool: torch.Tensor,
        repa_layer: int = -1,
        attn_mask: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        latent_x = self.patchify(latent)
        B, N, C = latent_x.size()

        if cond_mask is not None:
            cond_pool = torch.where(cond_mask, self.null_cond_vector.unsqueeze(0).expand_as(cond_pool), cond_pool)

        text_x = self.embedding_proj(cond)

        condition_input = self.time_embedding(timesteps) + self.pool_proj(cond_pool)

        attn_mask = ((
            (torch.cat(
                (torch.ones(B, N, device=latent.device).bool(), attn_mask), dim=-1
            ) * cond_mask).view(B, 1, 1, -1)).bool()
            if attn_mask is not None
            else None
        ) 

        for layer in self.multi_stream_layers:
            latent_x, text_x = layer(
                latent_x,
                text_x,
                condition_input,
                attn_mask=attn_mask,
            )

        x = torch.cat((latent_x, text_x), dim=1)

        repa_out = None

        
        for i, layer in enumerate(self.single_stream_layers):
            x = layer(x, condition_input, attn_mask=attn_mask)
            if i == repa_layer:
                repa_out = x[:, :N, :]

        x = self.final_ln(x)

        if repa_out is None:
            return self.depatchify(x, N)
        else:
            return self.depatchify(x, N), repa_out
