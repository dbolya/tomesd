# Token Merging for Stable Diffusion

Using nothing but pure python and pytorch, ToMe for SD speeds up diffusion by merging _redundant_ tokens.

![ToMe for SD applied on a 2048x2048 image.](https://raw.githubusercontent.com/dbolya/tomesd/main/examples/assets/teaser.jpg)

This is the official implementation of **ToMe for SD** from our short paper:  
**[Token Merging for Fast Stable Diffusion](https://arxiv.org/abs/2303.17604)**  
[Daniel Bolya](https://dbolya.github.io), [Judy Hoffman](https://faculty.cc.gatech.edu/~judy/)  
_[GitHub](https://github.com/dbolya/tomesd)_ | _[arXiv](https://arxiv.org/abs/2303.17604)_ | _[BibTeX](#citation)_

ToMe for SD is an extension of the original **ToMe**:  
**[Token Merging: Your ViT but Faster](https://arxiv.org/abs/2210.09461)**  
[Daniel Bolya](https://dbolya.github.io), 
[Cheng-Yang Fu](http://www.cs.unc.edu/~cyfu/),
[Xiaoliang Dai](https://sites.google.com/view/xiaoliangdai/),
[Peizhao Zhang](https://research.facebook.com/people/zhang-peizhao/),
[Christoph Feichtenhofer](https://feichtenhofer.github.io/),
[Judy Hoffman](https://faculty.cc.gatech.edu/~judy/)  
_[ICLR '23 Oral (Top 5%)](https://openreview.net/forum?id=JroZRaRw7Eu)_ | _[GitHub](https://github.com/facebookresearch/ToMe)_ | _[arXiv](https://arxiv.org/abs/2210.09461)_ | _[Blog](https://research.facebook.com/blog/2023/2/token-merging-your-vit-but-faster/)_ | _[BibTeX](https://github.com/facebookresearch/ToMe#citation)_

**Note:** this extension of ToMe is not affiliated in any way with Meta.


## What is ToMe for SD?
![A diffusion block with ToMe applied and the resulting images at different merge ratios.](https://raw.githubusercontent.com/dbolya/tomesd/main/examples/assets/method.jpg)

Token Merging (**ToMe**) speeds up transformers by _merging redundant tokens_, which means the transformer has to do _less work_. We apply this to the underlying transformer blocks in Stable Diffusion in a clever way that minimizes quality loss while keeping most of the speed-up and memory benefits. ToMe for SD _doesn't_ require training and should work out of the box for any Stable Diffusion model.

**Note:** this is a lossy process, so the image _will_ change, ideally not by much. Here are results with [FID](https://github.com/mseitzer/pytorch-fid) scores vs. time and memory usage (lower is better) when using Stable Diffusion v1.5 to generate 512x512 images of ImageNet-1k classes on a 4090 GPU with 50 PLMS steps using fp16:

| Method                      | r% | FID ↓  | Time (s/im) ↓            | Memory (GB/im) ↓        |
|-----------------------------|----|:------|:--------------------------|:------------------------|
| Baseline _(Original Model)_ | 0  | 33.12 | 3.09                      | 3.41                    |
| w/ **ToMe for SD**        | 10 | 32.86 | 2.56 (**1.21x** _faster_) | 2.99 (**1.14x** _less_) |
|                             | 20 | 32.86 | 2.29 (**1.35x** _faster_) | 2.17 (**1.57x** _less_) |
|                             | 30 | 32.80 | 2.06 (**1.50x** _faster_) | 1.71 (**1.99x** _less_) |
|                             | 40 | 32.87 | 1.85 (**1.67x** _faster_) | 1.26 (**2.71x** _less_) |
|                             | 50 | 33.02 | 1.65 (**1.87x** _faster_) | 0.89 (**3.83x** _less_) |
|                             | 60 | 33.37 | 1.52 (**2.03x** _faster_) | 0.60 (**5.68x** _less_) |

Even with more than half of the tokens merged (60%!), ToMe for SD still produces images close to the originals, while being _**2x** faster_ and using _**~5.7x** less memory_. Moreover, ToMe is not another efficient reimplementation of transformer modules. Instead, it actually _reduces_ the total work necessary to generate an image, so it can function _in conjunction_ with efficient implementations (see [Usage](#tome--xformers--flash-attn--torch-20)).

## News
 - **[2023.04.02]** ToMe for SD is now available via pip as [tomesd](https://pypi.org/project/tomesd/). Thanks @mkshing!
 - **[2023.03.31]** ToMe for SD now supports [Diffusers](https://github.com/huggingface/diffusers). Thanks @JunnYu and @ExponentialML!
 - **[2023.03.30]** Initial release.

See the [changelog](CHANGELOG.md) for more details.


## Supported Environments

This repo includes code to patch an existing Stable Diffusion environment. Currently, we support the following implementations:
 - [x] [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion)
 - [x] [Stable Diffusion v1](https://github.com/runwayml/stable-diffusion)
 - [x] [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
 - [x] [Diffusers](https://github.com/huggingface/diffusers)
 - [ ] And potentially others

**Note:** This also supports most downstream UIs that use these repositories.


## Installation

ToMe for SD requires ``pytorch >= 1.12.1`` (for `scatter_reduce`), which you can get from [here](https://pytorch.org/get-started/locally/). Then after installing your choice of stable diffusion environment ([supported environments](#supported-environments)), use the corresponding python environment to install ToMe for SD:

```bash
pip install tomesd
```

### Installing from source
If you'd like to install from source to get the latest updates, clone the repository:
```bash
git clone https://github.com/dbolya/tomesd
cd tomesd
```
Then set up the tomesd package with:
```bash
python setup.py build develop
```
That's it! ToMe for SD is implemented in pure python, no CUDA compilation required. 🙂


## Usage
Apply ToMe for SD to any Stable Diffusion model with
```py
import tomesd

# Patch a Stable Diffusion model with ToMe for SD using a 50% merging ratio.
# Using the default options are recommended for the highest quality, tune ratio to suit your needs.
tomesd.apply_patch(model, ratio=0.5)

# However, if you want to tinker around with the settings, we expose several options.
# See docstring and paper for details. Note: you can patch the same model multiple times.
tomesd.apply_patch(model, ratio=0.9, sx=4, sy=4, max_downsample=2) # Extreme merging, expect diminishing returns
```
See above for what speeds and memory savings you can expect with different ratios.
If you want to remove the patch later, simply use `tomesd.remove_patch(model)`.

### Example
To apply ToMe to the txt2img script of SDv2 or SDv1 for instance, add the following to [this line](https://github.com/Stability-AI/stablediffusion/blob/fc1488421a2761937b9d54784194157882cbc3b1/scripts/txt2img.py#L220) (SDv2) or [this line](https://github.com/runwayml/stable-diffusion/blob/08ab4d326c96854026c4eb3454cd3b02109ee982/scripts/txt2img.py#L241) (SDv1):
```py
import tomesd
tomesd.apply_patch(model, ratio=0.5)
```
That's it! More examples and demos coming soon (_hopefully_).  
**Note:** You may not see the full speed-up for the first image generated (as pytorch sets up the graph). Since ToMe for SD uses random processes, you might need to set the seed every batch if you want consistent results.

### Diffusers
ToMe can also be used to patch a 🤗 Diffusers Stable Diffusion pipeline:
```py
import torch, tomesd
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# Apply ToMe with a 50% merging ratio
tomesd.apply_patch(pipe, ratio=0.5) # Can also use pipe.unet in place of pipe here

image = pipe("a photo of an astronaut riding a horse on mars").images[0]
image.save("astronaut.png")
```
You can remove the patch with `tomesd.remove_patch(pipe)`.

### ToMe + xformers / flash attn / torch 2.0
Since ToMe only affects the forward function of the block, it should support most efficient transformer implementations out of the box. Just apply the patch as normal!

**Note:** when testing with xFormers, I observed the most speed-up with ToMe when using _big_ images (i.e., 2048x2048 in the parrot example above). You can get even more speed-up with more aggressive merging configs, but quality obviously suffers. For the result above, I had each method img2img from the same 512x512 res image (i.e., I only applied ToMe during the second pass of "high res fix") and used the default config with 60% merging. Also, the memory benefits didn't stack with xFormers (efficient attention already takes care of memory concerns).


## Citation

If you use ToMe for SD or this codebase in your work, please cite:
```
@article{bolya2023tomesd,
  title={Token Merging for Fast Stable Diffusion},
  author={Bolya, Daniel and Hoffman, Judy},
  journal={arXiv},
  year={2023}
}
```
If you use ToMe in general please cite the original work:
```
@inproceedings{bolya2023tome,
  title={Token Merging: Your {ViT} but Faster},
  author={Bolya, Daniel and Fu, Cheng-Yang and Dai, Xiaoliang and Zhang, Peizhao and Feichtenhofer, Christoph and Hoffman, Judy},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```