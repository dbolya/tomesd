import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # src, dst = x[..., ::2, :], x[..., 1::2, :]
        # n, t1, c = src.shape
        # unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)#dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        # return torch.cat([unm, dst], dim=1)
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_add(-2, dst_idx.expand(B, r, C), src)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # unm_len = unm_idx.shape[1]
        # unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        # _, _, c = unm.shape

        # src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # # Combine back to the original shape
        # out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        # out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        # out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        # out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        # return out
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out
        
    return merge, unmerge
