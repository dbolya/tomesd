import torch

from typing import Dict, Any, Tuple, Optional
from torch import nn

from .utils import isinstance_str, init_generator
from timm.models.vision_transformer import Attention

from .utils import parse_r
from .merge import merge_wavg, merge_source, bipartite_soft_matching, bipartite_soft_matching_random2d

class ToMeResidualAttentionBlock(nn.Module):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """        

    def forward(self, q_x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        
        x_attn, metric = self.attn(self.ln_1(q_x), attn_size)
        x = q_x + self.ls_1(x_attn)
        
        r = self._tome_info["r"].pop(0) 
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
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, 
        size: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        need_weights = False,
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
            self._tome_info["r"] = parse_r(len(self.transformer.resblocks), self.r) #self._tome_info["ratio"]
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


def patch_openclip(
    model, ratio, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe the OpenCLIP model.
    """
    vision_model = model.visual
    ToMeVisionTransformer = make_tome_class(vision_model.__class__)

    vision_model.__class__ = ToMeVisionTransformer
    vision_model.r = ratio
    vision_model._tome_info = {
        "ratio": ratio,
        "r": vision_model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(vision_model, "dist_token") and vision_model.dist_token is not None:
        vision_model._tome_info["distill_token"] = True

    for i, resblock in enumerate(vision_model.transformer.resblocks):     
        resblock.__class__ = ToMeResidualAttentionBlock
        resblock._tome_info = vision_model._tome_info
        attn = ToMeAttention(resblock.attn.embed_dim, resblock.attn.num_heads, qkv_bias=True)
        _, device = convert_attention_block(resblock.attn, attn)
        attn = attn.to(device)
        resblock.attn = attn
        