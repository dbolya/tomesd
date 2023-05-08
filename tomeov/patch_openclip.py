import torch
import math
from typing import Type, Dict, Any, Tuple, Callable, Optional
from collections import OrderedDict
from torch import nn

from . import merge
from .utils import isinstance_str, init_generator
from timm.models.vision_transformer import Attention
from open_clip.transformer import ResidualAttentionBlock, LayerNorm, LayerScale

from tomeov.utils import parse_r


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).
    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)


        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size

def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


class ToMeResidualAttentionBlock(nn.Module):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    # def __init__(
    #         self,
    #         d_model: int,
    #         n_head: int,
    #         mlp_ratio: float = 4.0,
    #         ls_init_value: float = None,
    #         act_layer: Callable = torch.nn.GELU,
    #         norm_layer: Callable = LayerNorm,
    #         is_cross_attention: bool = False,
    # ):
    #     super().__init__()
        
    #     self.ln_1 = norm_layer(d_model)
    #     self.attn = ToMeAttention(dim=d_model, num_heads=n_head, qkv_bias=True)
    #     self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
    #     if is_cross_attention:
    #         self.ln_1_kv = norm_layer(d_model)

    #     self.ln_2 = norm_layer(d_model)
    #     mlp_width = int(d_model * mlp_ratio)
    #     self.mlp = nn.Sequential(OrderedDict([
    #         ("c_fc", nn.Linear(d_model, mlp_width)),
    #         ("gelu", act_layer()),
    #         ("c_proj", nn.Linear(mlp_width, d_model))
    #     ]))
    #     self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        

    def forward(self, q_x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        #q_x = q_x.permute(1, 0, 2)
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        
        
        
        x_attn, metric = self.attn(self.ln_1(q_x), attn_size)
        # t_x = self.ln_1(q_x)
        
        # x_attn, metric = self.attn(t_x.permute(1, 0, 2), attn_size)
        # x_attn = x_attn.permute(1, 0, 2)
        
        x = q_x + self.ls_1(x_attn)
        
        r = int(q_x.shape[1] * self._tome_info["ratio"]) #self._tome_info["r"].pop(0)
        print("r: ", r)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x#.permute(1, 0, 2)


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, 
        size: torch.Tensor = None,
        attn_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)
    
def convert_attention_block(
    src: nn.MultiheadAttention, dst: ToMeAttention
) -> Tuple[ToMeAttention, torch.device]:
    src_state_dict = src.state_dict()
    dst_state_dict = dst.state_dict()
    src_to_dst_keys = [
        ("in_proj_weight", "qkv.weight"),
        ("in_proj_bias", "qkv.bias"),
        ("out_proj.weight", "proj.weight"),
        ("out_proj.bias", "proj.bias"),
    ]

    # for key, data in src_state_dict.items():
    #     print(key, data.dtype)

    for src_key, dst_key in src_to_dst_keys:
        dst_state_dict[dst_key] = src_state_dict[src_key]
    dst.load_state_dict(dst_state_dict)
    src_device = src_state_dict["in_proj_weight"].device
    return dst.to(src_device), src_device


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.transformer.resblocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
            if self.input_patchnorm:
                # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
                x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
                x = x.permute(0, 2, 4, 1, 3, 5)
                x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
                x = self.patchnorm_pre_ln(x)
                x = self.conv1(x)
            else:
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            # class embeddings and positional embeddings
            x = torch.cat(
                [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)

            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            x = self.patch_dropout(x)
            x = self.ln_pre(x)

            x = self.transformer(x)

            if self.attn_pool is not None:
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
            else:
                pooled, tokens = self._global_pool(x)
                pooled = self.ln_post(pooled)

            if self.proj is not None:
                pooled = pooled @ self.proj

            if self.output_tokens:
                return pooled, tokens
            
            return pooled

    return ToMeVisionTransformer





def apply_openclip_patch(
    model, ratio: float, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.
    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.
    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "ratio": ratio,
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for i, resblock in enumerate(model.transformer.resblocks):     
        resblock.__class__ = ToMeResidualAttentionBlock
        resblock._tome_info = model._tome_info
        attn = ToMeAttention(resblock.attn.embed_dim, resblock.attn.num_heads, qkv_bias=True)
        _, device = convert_attention_block(resblock.attn, attn)
        attn = attn.to(device)
        resblock.attn = attn
        