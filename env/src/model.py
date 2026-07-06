import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import T5EncoderModel, CLIPTextModelWithProjection


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, x.size()[1:])


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

        self.rope = RoPE(d_model=d_model)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        Q = norm(self.rope(self.wq(x)).view(B, T, self.n_heads, -1).transpose(1, 2))
        K = norm(self.rope(self.wk(x)).view(B, T, self.n_heads, -1).transpose(1, 2))
        V = self.wv(x).view(B, T, self.n_heads, -1).transpose(1, 2)

        attn_scores = (
            F.scaled_dot_product_attention(Q, K, V).permute(0, 2, 1, 3).contiguous()
        )

        return self.wo(attn_scores.view(B, T, C))


class CrossAttention(MultiHeadAttention):
    def __init__(self, d_model: int, n_heads: int):
        """
        CrossAttention
        d_model (int): model dimension
        n_heads (int): number of heads to split d_model across

        input size: (B, N, C), (B, T, C)
        output size: (B, N, C)
        """
        super().__init__(d_model=d_model, n_heads=n_heads)

    def forward(self, img_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        B, N, C = img_x.size()
        T = text_x.size(1)
        Q = norm(self.rope(self.wq(img_x)).view(B, N, self.n_heads, -1).transpose(1, 2))
        K = norm(
            self.rope(self.wk(text_x)).view(B, T, self.n_heads, -1).transpose(1, 2)
        )
        V = self.wv(text_x).view(B, T, self.n_heads, -1).transpose(1, 2)

        attn_scores = (
            F.scaled_dot_product_attention(Q, K, V).permute(0, 2, 1, 3).contiguous()
        )

        # attn(Q, K, V) -> softmax(Q K^T)V
        # (B, H, N, d) x (B, H, d, T) -> (B, H, N, T)
        # (B, H, N, T) x (B, H, T, d) -> (B, H, N, d) -> (B, N, H, d)

        return self.wo(attn_scores.view(B, N, C))


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
        output size: (B, c, W, H)
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
        # reshape to B, c, W, H
        return (
            self.proj(x)[:, :N, :]
            .view(B, N, self.n_channels, self.patch_size, self.patch_size)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, self.n_channels, self.w, self.h)
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
    def __init__(self, d_model: int, cache_positions: int = 1024, base: float = 10000):
        """
        RoPE
        d_model (int): model dimension
        cache_positions (int): number of precomputed sines/cosines to store
        base (float): base used for calculating thetas

        input size: (B, T, C)
        output size: (B, T, C)
        """

        super().__init__()
        self.d_model = d_model
        self.cache_positions = cache_positions
        self.base = base

        self.thetas = torch.empty(d_model // 2)

        for i in range(d_model // 2):
            self.thetas[i] = self.base ** (-2 * (i - 1) / d_model)

        self.thetas = torch.repeat_interleave(self.thetas, 2, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        odds = x[:, :, 1::2]
        evens = x[:, :, ::2]

        rotated = torch.empty_like(x, device=x.device)

        ms = torch.arange(T, device=x.device)

        angles = torch.einsum("t, d -> td", ms, self.thetas.to(x.device)).unsqueeze(0)
        sines = torch.sin(angles)
        cosines = torch.cos(angles)

        rotated[:, :, 1::2] = -evens
        rotated[:, :, ::2] = odds

        return x * cosines[:, :T, :] + rotated * sines[:, :T, :]


class MultiStreamDiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_index: int):
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
        self.layer_index = layer_index

        self.cross_attention = CrossAttention(d_model=d_model, n_heads=n_heads)

        self.mlp = SwiGLU(d_in=d_model, d_h=d_model * 4, d_out=d_model)

    def forward(
        self, img_x: torch.Tensor, text_x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor]:
        img_x = (
            img_x
            + self.cross_attention(
                img_x * condition[:, self.layer_index, 0, None, None]
                + condition[:, self.layer_index, 1, None, None],
                text_x * condition[:, self.layer_index, 2, None, None]
                + condition[:, self.layer_index, 3, None, None],
            )
            * condition[:, self.layer_index, 4, None, None]
        )
        img_x = (
            img_x
            + self.mlp(
                norm(img_x) * condition[:, self.layer_index, 5, None, None]
                + condition[:, self.layer_index, 6, None, None]
            )
            * condition[:, self.layer_index, 7, None, None]
        )
        return img_x, text_x


class SingleStreamDitBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_index: int):
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
        self.layer_index = layer_index

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        self.mlp = SwiGLU(d_model, d_model * 4, d_model)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = (
            x
            + self.attention(
                norm(x) * condition[:, self.layer_index, 0, None, None]
                + condition[:, self.layer_index, 1, None, None]
            )
            * condition[:, self.layer_index, 2, None, None]
        )
        x = (
            x
            + self.mlp(
                norm(x) * condition[:, self.layer_index, 3, None, None]
                + condition[:, self.layer_index, 4, None, None]
            )
            * condition[:, self.layer_index, 5, None, None]
        )
        return x


class DiT(nn.Module):
    def __init__(
        self,
        encoder_model: T5EncoderModel,
        clip_encoder_model: CLIPTextModelWithProjection,
        d_model: int,
        n_heads: int,
        n_layers_multi_stream: int,
        n_layers_single_stream: int,
        patch_size: int,
        w: int,
        h: int,
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
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels

        self.encoder = encoder_model
        self.clip_encoder = clip_encoder_model
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        for p in self.clip_encoder.parameters():
            p.requires_grad = False

        self.embedding_proj = nn.Linear(
            self.encoder.config.d_model, d_model, bias=False
        )

        self.patchify = ConvPatchify(
            d_model=d_model, patch_size=patch_size, n_channels=n_channels
        )

        self.time_embedding = SinusoidalEmbedding(
            n_embeddings=n_timesteps, d_model=self.clip_encoder.config.hidden_size
        )

        self.multi_stream_condition_mlp = MLP(
            self.clip_encoder.config.hidden_size, d_model, n_layers_multi_stream * 8
        )
        nn.init.zeros_(self.multi_stream_condition_mlp.l2.weight)
        nn.init.zeros_(self.multi_stream_condition_mlp.l2.bias)

        self.single_stream_condition_mlp = MLP(
            self.clip_encoder.config.hidden_size, d_model, n_layers_single_stream * 6
        )
        nn.init.zeros_(self.single_stream_condition_mlp.l2.weight)
        nn.init.zeros_(self.single_stream_condition_mlp.l2.bias)

        self.multi_stream_layers = nn.ModuleList(
            [
                MultiStreamDiTBlock(d_model=d_model, n_heads=n_heads, layer_index=i)
                for i in range(n_layers_multi_stream)
            ]
        )

        self.single_stream_layers = nn.ModuleList(
            [
                SingleStreamDitBlock(d_model=d_model, n_heads=n_heads, layer_index=i)
                for i in range(n_layers_single_stream)
            ]
        )

        self.final_ln = nn.LayerNorm(d_model)

        self.depatchify = Depatchify(
            d_model=d_model, patch_size=patch_size, w=w, h=h, n_channels=n_channels
        )
        nn.init.xavier_normal_(self.depatchify.proj.weight)

    def forward(
        self, latent: torch.Tensor, tokens: torch.Tensor, clip_tokens: torch.Tensor, timesteps: torch.Tensor, repa_layer: int = -1,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        latent_x = self.patchify(norm(latent))
        B, N, C = latent_x.size()

        with torch.no_grad():
            encoder_hiddens = self.encoder(tokens, output_hidden_states=False).last_hidden_state
            text_pool = self.clip_encoder(clip_tokens, output_hidden_states=False).text_embeds

        text_x = self.embedding_proj(norm(encoder_hiddens))
        condition_input = norm(text_pool) + self.time_embedding(timesteps)
        multi_stream_condition = self.multi_stream_condition_mlp(condition_input).view(
            B, self.n_layers_multi_stream, -1
        )

        single_stream_condition = self.single_stream_condition_mlp(
            condition_input
        ).view(B, self.n_layers_single_stream, -1)

        for layer in self.multi_stream_layers:
            latent_x, text_x = layer(latent_x, text_x, multi_stream_condition)

        x = torch.cat((latent_x, text_x), dim=1)

        repa_out = None
        for i, layer in enumerate(self.single_stream_layers):
            x = layer(x, single_stream_condition)
            if i == repa_layer:
                repa_out = x[:, :N, :]

        # print(x)
        # print(self.final_ln(x))
        if repa_out is None:
            return self.depatchify(x, N) # self.final_ln(x)
        else:
            return self.depatchify(x, N), repa_out
