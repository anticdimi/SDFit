import copy
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import omegaconf as omconf
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from pytorch3d.transforms import rotation_6d_to_matrix
from torch import nn


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)

    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1).permute(1, 0, 2).reshape(1, -1, 2).repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,), replace=False)
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def smooth_mask_edges(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to the mask to smooth its edges.
    Args:
        mask: (H, W) mask
        blur_size: size of the Gaussian blur kernel
    Returns:
        Smoothed mask
    """
    smoothed_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    smoothed_mask = (smoothed_mask > 0).astype(np.uint8)  # Binarize the mask again
    return smoothed_mask


def dilate_mask(mask: np.ndarray, dilation_size: int = 1) -> np.ndarray:
    """
    Expand the binary mask by a given number of pixels.
    Args:
        mask: (H, W) binary mask
        dilation_size: number of pixels to dilate the mask
    Returns:
        expanded_mask: (H, W) expanded binary mask
    """
    kernel = np.ones((2 * dilation_size + 1, 2 * dilation_size + 1), np.uint8)
    expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded_mask.astype(bool)


def normalize_depth(
    depth_image: torch.Tensor,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> torch.Tensor:
    # Identify non-zero values (object) and compute min/max
    non_zero_mask = depth_image != 0
    object_values = depth_image[non_zero_mask]
    min_depth = torch.min(object_values).squeeze()
    max_depth = torch.max(object_values).squeeze()

    # Normalize based on non-zero values
    normalized_depth = (depth_image - min_depth) / (max_depth - min_depth)

    # Apply new min and max
    normalized_depth = normalized_depth * (new_max - new_min) + new_min

    return normalized_depth


# ! Adapted from: https://github.com/lllyasviel/ControlNet/blob/main/annotator/midas/__init__.py#L33
def normals_from_depth_sobel(
    depth_np: np.ndarray,
    bg_mask: np.ndarray = None,
    ksize: int = 3,
    smoothing: bool = False,
    expand_mask: bool = False,
) -> np.ndarray:
    """
    Compute the normal vectors from a depth image.
    Args:
        depth_image: (H, W) depth image
    Returns:
        normals: (H, W, 3) normal vectors in the range [0, 255]
    """
    if smoothing:
        depth_np = cv2.GaussianBlur(depth_np, (ksize, ksize), 0)

    x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
    y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
    z = np.ones_like(x) * np.pi * 2.0

    if bg_mask is not None:
        if expand_mask:
            bg_mask = dilate_mask(bg_mask)
            bg_mask = smooth_mask_edges(bg_mask)
        x[bg_mask.squeeze()] = 0
        y[bg_mask.squeeze()] = 0
        z[bg_mask.squeeze()] = 0  # Ensure z is zeroed out where mask is invalid

    normal = np.stack([x, y, z], axis=2)
    normal_magnitude = np.sum(normal**2.0, axis=2, keepdims=True) ** 0.5
    normal_magnitude[normal_magnitude == 0] = 1  # Avoid division by zero
    normal /= normal_magnitude
    normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]
    return normal_image


class EdgeDetector(nn.Module):
    def __init__(self):
        super(EdgeDetector, self).__init__()
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.edge_filter.weight.data = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def forward(self, input_mask):
        # Apply the edge detection filter
        edges = self.edge_filter(input_mask)

        # Threshold the edges to get a binary contour mask
        contour_mask = F.relu(edges)
        contour_mask = (contour_mask - contour_mask.min()) / (
            contour_mask.max() - contour_mask.min()
        )
        return contour_mask


################################## * Geometry helpers ##################################
def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.view(-1, 3, 3)
    a1 = matrix[:, :, 0]
    a2 = matrix[:, :, 1]
    # a3 = matrix[:, :, 2]
    return torch.stack((a1, a2), dim=-1).view(-1, 6)


def calculate_centroid(mask):
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.nelement() == 0:
        return torch.tensor([mask.shape[0] / 2, mask.shape[1] / 2], device=mask.device)
    weights = mask[coords[:, 0], coords[:, 1]]
    # Ensure no in-place operations modify 'coords' or 'weights'
    centroid = torch.sum(coords * weights.unsqueeze(1), dim=0) / torch.sum(weights)
    return centroid


def apply_rigid_transform(
    points: torch.Tensor,
    rotation_matrix: torch.Tensor,
    translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    scale: torch.Tensor = torch.tensor([1.0, 1.0, 1.0]),
) -> torch.Tensor:
    """
    Apply a rigid transformation to a set of points.
    points: (N, 3) tensor
    rotation_matrix: (3, 3) tensor
    translation: (3,) tensor
    scale: (3,) tensor
    """
    device = points.device
    assert points.shape[1] == 3, f"Invalid shape {points.shape}"
    assert rotation_matrix.shape == (3, 3), f"Invalid shape {rotation_matrix.shape}"
    assert translation.shape == (3,), f"Invalid shape {translation.shape}"
    assert scale.shape == (3,), f"Invalid shape {scale.shape}"
    assert not points.isnan().any(), "There are NaNs in the input points"
    assert not rotation_matrix.isnan().any(), "There are NaNs in the input rotation matrix"
    assert not translation.isnan().any(), "There are NaNs in the input translation vector"
    assert not scale.isnan().any(), "There are NaNs in the input scale vector"

    transformation_matrix = torch.eye(4).to(device).float()
    scale_matrix = torch.diag(scale).to(device)
    transformation_matrix[:3, :3] = scale_matrix @ rotation_matrix.to(device)
    transformation_matrix[:3, 3] = translation.to(device)

    # Add homogeneous coordinate to points
    points = torch.cat((points, torch.ones((points.shape[0], 1)).to(device)), dim=1)
    # Apply transformation
    transformed_points = torch.matmul(points, transformation_matrix.T)

    # Remove homogeneous coordinate
    transformed_points = transformed_points[:, :3]

    return transformed_points


def normalize_points_to_unit_cube(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize a set of 3D points to be within the unit cube [-0.5, 0.5].

    Parameters:
    points (torch.Tensor): A tensor of shape (N, 3) representing N 3D points.

    Returns:
    torch.Tensor: A tensor of shape (N, 3) with the normalized points.
    """
    # Compute the minimum and maximum values along each axis
    min_vals = torch.min(points, dim=0).values
    max_vals = torch.max(points, dim=0).values

    # Compute the center of the bounding box
    center = (min_vals + max_vals) / 2.0

    # Translate points to be centered around the origin
    translated_points = points - center

    # Compute the scale factor to fit points within the unit cube
    max_side_length = max(max_vals - min_vals)

    # Normalize points to fit within the unit cube [-0.5, 0.5]
    normalized_points = translated_points / max_side_length

    return normalized_points


def rot_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """
    Convert a 6D representation to a 3x3 rotation matrix.
    Args:
        rotation: (6, ) 6D representation
    Returns:
        (3, 3) rotation matrix
    """
    assert rotation.shape[0] == 6, f"Invalid shape {rotation.shape}"
    assert not rotation.isnan().any(), "There are NaNs in the input matrix"

    return rotation_6d_to_matrix(rotation[None, ...])[0]


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)
    normalized = tensor / (norm + 1e-8)
    return normalized


def standardize(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.numel() > 0, "Input tensor is empty"

    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # If all elements are the same, return a zero tensor to avoid division by epsilon
    if tensor_max == tensor_min:
        return torch.zeros_like(tensor)

    standardized = (tensor - tensor_min) / (tensor_max - tensor_min)
    return standardized


############################################################################################
################################## * Type helpers/casters ##################################
METHOD_KEYS = ["update", "pop", "to", "to_dict", "detach", "clone", "cpu", "cuda"]


class EasierDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and k not in METHOD_KEYS:
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x) if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, EasierDict):
            value = EasierDict(value)
        super(EasierDict, self).__setattr__(name, value)
        super(EasierDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def clone(self):
        return EasierDict(copy.deepcopy(self.to_dict()))

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasierDict, self).pop(k, *args)

    def cpu(self) -> "EasierDict":
        return self.to("cpu")

    def cuda(self) -> "EasierDict":
        return self.to("cuda")

    def detach(self) -> "EasierDict":
        for key in self:
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].detach()
        return self

    def to(self, device: str | torch.device) -> "EasierDict":
        for key in self:
            if hasattr(self[key], "to") and not isinstance(self[key], (str, bytes)):
                self[key] = self[key].to(device)
        return self

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.items():
            if isinstance(v, EasierDict):
                d[k] = v.to_dict()
            elif isinstance(v, (list, tuple)):
                d[k] = type(v)(EasierDict(x).to_dict() if isinstance(x, dict) else x for x in v)
            else:
                d[k] = v
        return d


def to_tensor(arr: list | np.ndarray | torch.Tensor | Any, **kwargs) -> torch.Tensor:
    if kwargs.get("dtype") is None:
        kwargs["dtype"] = torch.float32
    # Ensure the input is of the correct type
    if not isinstance(arr, (list, np.ndarray, torch.Tensor, np.generic, int, float)):
        raise TypeError("Input must be a list, np.ndarray, torch.Tensor or a number.")

    # Handle single number input, including numpy scalars
    if isinstance(arr, (int, float, np.generic)):
        return torch.tensor(arr, **kwargs)

    if isinstance(arr, list):
        arr = np.array(arr)

    if isinstance(arr, torch.Tensor):
        return arr.to(**kwargs)

    return torch.tensor(arr, **kwargs)


############################################################################################
################################## * FS/IO helpers ##################################
def mkdir(path: Path | str) -> None:
    """
    Create a directory if it does not exist.
    Args:
        path (Path | str): The path to the directory to create.
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_mesh(fname: str | Path) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fname, Path):
        fname = Path(fname)
    ext = fname.suffix[1:]
    obj_info = trimesh.load(fname, file_type=ext, process=False)
    if type(obj_info) is trimesh.Scene:
        geo_keys = list(obj_info.geometry.keys())
        total_vert = []
        total_faces = []
        for gk in geo_keys:
            cur_geo = obj_info.geometry[gk]
            cur_vert = cur_geo.vertices.tolist()
            cur_face = np.array(cur_geo.faces.tolist()) + len(total_vert)
            total_vert += cur_vert
            total_faces += cur_face.tolist()
        return np.array(total_vert).astype("float32"), np.array(total_faces).astype("int32")
    else:
        return (
            np.array(obj_info.vertices).astype("float32"),
            np.array(obj_info.faces).astype("int32"),
        )


def save_mesh(
    v: np.ndarray,
    f: np.ndarray,
    fname: str | Path,
    v_colors: np.ndarray | None = None,
) -> None:
    if not isinstance(fname, Path):
        fname = Path(fname)
    ext = fname.suffix[1:]
    mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=v_colors, process=False)
    mesh.export(fname, file_type=ext)


def find_files(directory: str | Path, pattern: str) -> list[Path]:
    """
    Recursively search for files matching the given pattern in the specified directory.

    :param directory: The directory to start the search from.
    :param pattern: The pattern to match files.
    :return: A list of Paths to the files that match the pattern.
    """
    if not isinstance(directory, Path):
        directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory")

    matching_files = []

    for item in directory.iterdir():
        if item.is_dir():
            # Recurse into the directory
            matching_files.extend(find_files(item, pattern))
        elif item.is_file() and item.match(pattern):
            matching_files.append(item)

    return matching_files


def save_image(image: np.ndarray | Image.Image, path: str | Path, mode: str = "RGB") -> None:
    if np.asarray(image).max() <= 1:
        image *= 255
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8), mode=mode)
    elif isinstance(image, torch.Tensor):
        image = Image.fromarray(image.cpu().numpy().astype(np.uint8), mode=mode)
    assert isinstance(image, Image.Image), f"Invalid image type {type(image)}"
    image.save(str(path))


def load_image(
    path: Path | str,
    ret_np: bool = True,
    mode: str = "RGB",
) -> np.ndarray | Image.Image:
    """
    Load an image from disk.
    Returns a numpy array if ret_np is True, otherwise returns a PIL Image. Range [0, 255]
    """
    assert (img := Image.open(str(path))) is not None, f"Could not load image {path}"
    img = img.convert(mode)
    return np.asarray(img) if ret_np else img


def load_config() -> dict:
    dot_list = sys.argv[1:]
    def_cfg_p = sys.argv[1]
    assert Path(def_cfg_p).exists(), f"Config file {def_cfg_p} does not exist"

    default_cfg = omconf.OmegaConf.load(def_cfg_p)
    for k, v in omconf.OmegaConf.from_dotlist(dot_list).items():
        omconf.OmegaConf.update(default_cfg, k, v, merge=True)
    return default_cfg


def dump_yaml(path: str | Path, data: dict | EasierDict) -> None:
    import yaml

    if isinstance(data, dict):
        yaml.dump(data, open(path, "w"), indent=4, default_flow_style=False)
    elif isinstance(data, omconf.DictConfig):
        omconf.OmegaConf.save(data, str(path))


############################################################################################
################################## * Misc stuff ##################################


# def get_timestamp():
#     return f'{datetime.now().strftime("%y%m%d-%H%M%S")}'


def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def fix_seeds(seed: int, **kwargs) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if kwargs.get("cudnn", False):
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def human_readable_time(seconds):
    """Convert seconds to a more readable format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
