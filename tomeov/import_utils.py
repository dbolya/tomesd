def is_diffusers_available():
    try:
        import stable_diffusion
        import utils
        return True
    except ImportError:
        return False

def is_openclip_available():
    try:
        import open_clip_torch
        return True
    except ImportError:
        return False

def is_timm_available():
    try:
        import timm
        return True
    except ImportError:
        return False
