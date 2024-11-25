import inspect
import re
import os
import yaml
import random
from typing import Callable, Any
import numpy as np
import torch


def seed_all(seed: int) -> None:
    """
    Seeds all RNGs for all used libraries
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Set deterministic backend for CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True


def mps_is_available() -> bool:
    """
    Analogous to `torch.cuda.is_available()` but for MPS
    """

    try:
        torch.ones(1).to("mps")
        return True
    except Exception:
        return False


def select_device() -> torch.device:
    """
    Returns the best available device
    """

    if mps_is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Adapted from
# https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/utils/dict.py
def print_dict(val, nesting: int = -4, start: bool = True) -> None:
    """Prints a nested dictionary."""

    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        # Deal with functions in print statements
        if callable(val):
            print(callable_to_string(val))
        else:
            print(val)


# Adapted from
# https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/utils/string.py
def callable_to_string(value: Callable) -> str:
    """Converts a callable object to a string.

    Args:
        value: A callable object.

    Raises:
        ValueError: When the input argument is not a callable object.

    Returns:
        A string representation of the callable object.
    """

    # Check if callable
    if not callable(value):
        raise ValueError(f"The input argument is not callable: {value}.")

    # Check if lambda function
    if value.__name__ == "<lambda>":
        # We resolve the lambda expression by checking the source code and extracting the line with lambda expression
        # We also remove any comments from the line
        lambda_line = inspect.getsourcelines(value)[0][0].strip().split("lambda")[1].strip().split(",")[0]
        lambda_line = re.sub(r"#.*$", "", lambda_line).rstrip()
        return f"lambda {lambda_line}"
    else:
        # Get the module and function name
        module_name = value.__module__
        function_name = value.__name__
        return f"{module_name}:{function_name}"


# Adapted from
# https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/utils/dict.py
def class_to_dict(obj: object) -> dict[str, Any]:
    """Converts an object into a dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj: An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Converted dictionary mapping.
    """

    # Check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")

    # Convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj

    # Convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # Disregard builtin attributes
        if key.startswith("__"):
            continue
        # Check if attribute is callable
        if callable(value):
            data[key] = callable_to_string(value)
        # Check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def dump_yaml(filename: str, data: dict | object) -> None:
    """Writes a dict or object to a YAML file."""

    if not filename.endswith("yaml"):
        filename += ".yaml"

    # Create directory if necessary
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert data into dictionary
    if not isinstance(data, dict):
        data = class_to_dict(data)

    # Save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
