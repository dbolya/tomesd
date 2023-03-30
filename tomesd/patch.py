import torch
import math
from typing import Type

from . import merge
from .utils import isinstance_str


def make_tome_block(
        block_class: Type[torch.nn.Module],
        ratio: float,
        max_downsample: int,
        merge_attn: bool,
        merge_crossattn: bool,
        merge_mlp: bool,
        sx: int, sy: int, no_rand: bool) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            original_h, original_w = self._tome_info["size"]
            original_tokens = original_h * original_w
            downsample = int(math.sqrt(original_tokens // x.shape[1]))

            if downsample <= max_downsample:
                w = original_w // downsample
                h = original_h // downsample
                r = int(x.shape[1] * ratio)
                m, u = merge.bipartite_soft_matching_random2d(x, w, h, sx, sy, r, no_rand)
            else:
                m, u = (merge.do_nothing, merge.do_nothing)

            m_a, u_a = (m, u) if merge_attn      else (merge.do_nothing, merge.do_nothing)
            m_c, u_c = (m, u) if merge_crossattn else (merge.do_nothing, merge.do_nothing)
            m_m, u_m = (m, u) if merge_mlp       else (merge.do_nothing, merge.do_nothing)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock



def make_tome_model(model_class):
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    
    This patches the forward function of the model only to get the current image size.
    Probably would be better off finding a way to get the size some other way.
    """

    if model_class.__name__ == "ToMeDiffusionModel":
        model_class = model_class._parent
    
    class ToMeDiffusionModel(model_class):
        # Save for later
        _parent = model_class

        def forward(self, *args, **kwdargs):
            self._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
            return super().forward(*args, **kwdargs)

    return ToMeDiffusionModel








def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Must divide the image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
        # Provided model not supported
        raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")

    diffusion_model = model.model.diffusion_model
    diffusion_model._tome_info = { "size": None, }
    diffusion_model.__class__ = make_tome_model(diffusion_model.__class__)

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            module.__class__ = make_tome_block(
                module.__class__, ratio, max_downsample,
                merge_attn, merge_crossattn, merge_mlp,
                sx, sy, not use_rand
            )
            module._tome_info = diffusion_model._tome_info

            # Something introduced in SD 2.0
            if not hasattr(module, "disable_self_attn"):
                module.disable_self_attn = False

    return model





def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """\
    
    for _, module in model.named_modules():
        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
        elif module.__class__.__name__ == "ToMeDiffusionModel":
            module.__class__ = module._parent
    
    return model