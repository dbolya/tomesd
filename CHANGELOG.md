# Changelog


## v0.1
 - **[2023.03.30]** Initial release.
 - **[2023.03.31]** Added support for more resolutions than multiples of 16. (Fixes #8)
 - **[2023.03.31]** Added support for diffusers (thanks @JunnYu and @ExponentialML)! (Fixes #1)

## v0.1.1
 - **[2023.04.01]** Rewrote how the model patching works to address some compatibility issues (e.g., ControlNet). (Fixes #9)

## v0.1.2
 - **[2023.04.03]** Added support for MPS devices (i.e., M1/M2 Macs). Thanks @brkirch! (Fixes #4)
   - **Note:** This fix still isn't perfect and may require some extra changes.
 - **[2023.04.04]** `use_rand` now forces itself off if the batch size is odd (meaning the prompted and unprompted images arent in the same batch). This should fix some artifacting without needing to tinker with the settings.


 ## v0.1.3
  - **[2023.04.24]** Random perturbations now use a separate rng so it doesn't affect the rest of the diffusion process. Thanks @alihassanijr!
  - **[2023.04.25]** Fixed an issue with the separate rng on mps devices. (Fixes #27)
  - **[2023.05.14]** Added fallback to CPU for non-supported devices for the separate rng generator.
  - **[2023.05.14]** Defined `use_ada_layer_norm_zero` just in case for older diffuser versions. (Fixes #20)