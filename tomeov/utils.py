import tempfile
import os
from pathlib import Path
from typing import Union, List, Tuple

import torch

from openvino._offline_transformations import apply_moc_transformations, compress_quantize_weights_transformation

from optimum.exporters.onnx import export_models, get_stable_diffusion_models_for_export
from optimum.intel import OVStableDiffusionPipeline
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)

def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.
    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def init_generator(device: torch.device):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu" or device.type == "mps": # MPS can use a cpu generator
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    raise NotImplementedError(f"Invalid/unsupported device. Expected `cpu`, `cuda`, or `mps`, got {device.type}.")


def _export_to_onnx(pipeline, save_dir):
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder

    unet.eval().cpu()
    vae.eval().cpu()
    text_encoder.eval().cpu()

    ONNX_WEIGHTS_NAME = "model.onnx"

    output_names = [
        os.path.join(DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
        os.path.join(DIFFUSION_MODEL_UNET_SUBFOLDER, ONNX_WEIGHTS_NAME),
        os.path.join(DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
        os.path.join(DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
    ]

    with torch.no_grad():
        models_and_onnx_configs = get_stable_diffusion_models_for_export(pipeline)
        pipeline.save_config(save_dir)
        export_models(
            models_and_onnx_configs=models_and_onnx_configs, output_dir=Path(save_dir), output_names=output_names
        )


def _export_to_openvino(pipeline, onnx_dir, save_dir):
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(
        model_id=onnx_dir,
        from_onnx=True,
        model_save_dir=save_dir,
        tokenizer=pipeline.tokenizer,
        scheduler=pipeline.scheduler,
        feature_extractor=pipeline.feature_extractor,
        compile=False,
    )
    apply_moc_transformations(ov_pipe.unet.model, cf=False)
    compress_quantize_weights_transformation(ov_pipe.unet.model)
    ov_pipe.save_pretrained(save_dir)

def export_diffusion_pipeline(pipeline, path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        _export_to_onnx(pipeline, tmpdirname)
        _export_to_openvino(pipeline, tmpdirname, Path(path))
