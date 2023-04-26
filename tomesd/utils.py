import torch


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
