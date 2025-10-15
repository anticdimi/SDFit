# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from collections import defaultdict

import nvdiffrast.torch as dr
import torch
from torch import nn

from ..utils import EasierDict


def standardize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.numel() > 0, "Input tensor is empty"

    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # If all elements are the same, return a zero tensor to avoid division by epsilon
    if tensor_max == tensor_min:
        return torch.zeros_like(tensor)

    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor


def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = torch.tan(fovy / 2)
    return torch.tensor(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, 1 / y, 0, 0],
            [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Creates a transformation matrix
    Args:
        - R: 3x3 array of a rotation matrix
        - t: 3x1 array of a translation vector
    Returns:
        - T: 4x4 Transformation matrix
    """
    T = torch.zeros(4, 4, device=R.device, dtype=R.dtype)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    T[3, 3] = 1
    return T


def xfm_points(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    """
    out = torch.matmul(
        torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=1.0),
        torch.transpose(matrix, 1, 2),
    )
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out


def to_homog(points: torch.Tensor) -> torch.Tensor:
    """Converts points to homogeneous coordinates."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


# ! Adapted from https://github.dev/vchoutas/smplify-x
class Camera(nn.Module):
    FOCAL_LENGTH = 136.0

    def __init__(
        self,
        rotation: torch.Tensor = None,
        translation: torch.Tensor = None,
        focal_length_x: float = None,
        focal_length_y: float = None,
        center: torch.Tensor = None,
        dtype=torch.float32,
        cam_type: str = "weak",
    ):
        super().__init__()
        self.dtype = dtype
        self.cam_type = cam_type
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer("zero", torch.zeros([1], dtype=dtype))

        if focal_length_x is None or isinstance(focal_length_x, float):
            focal_length_x = torch.tensor(
                self.FOCAL_LENGTH if focal_length_x is None else focal_length_x,
                dtype=dtype,
            )

        if focal_length_y is None or isinstance(focal_length_y, float):
            focal_length_y = torch.tensor(
                self.FOCAL_LENGTH if focal_length_y is None else focal_length_y,
                dtype=dtype,
            )

        self.register_parameter("focal_length_x", nn.Parameter(focal_length_x, requires_grad=True))
        self.register_parameter("focal_length_y", nn.Parameter(focal_length_y, requires_grad=True))

        if center is None:
            center = torch.zeros([2], dtype=dtype)
        self.register_buffer("center", center)

        if rotation is None:
            rotation = torch.eye(3, dtype=dtype)
        self.register_parameter("rotation", nn.Parameter(rotation, requires_grad=True))

        if translation is None:
            translation = torch.zeros([3], dtype=dtype)
        self.register_parameter("translation", nn.Parameter(translation, requires_grad=True))

    def get_transform(
        self,
        res: list[float],
        z_near: float = 0.1,
        z_far: float = 1000.0,
        device="cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proj_mtx = perspective(
            2 * torch.arctan(res[0] / (2 * self.focal_length_y)),
            res[1] / res[0],
            z_near,
            z_far,
            device=device,
        )
        # Treat stored rotation/translation as camera pose (camera -> world)
        # Convert to view matrix (world -> camera): R_v = R^T, t_v = -R^T t
        R_c2w = self.rotation
        t_c2w = self.translation
        R_w2c = R_c2w.transpose(0, 1)
        t_w2c = -R_w2c @ t_c2w
        mv = transform_mat(R_w2c, t_w2c)
        mvp = proj_mtx @ mv
        return mv.to(device), mvp.to(device)


# ! Mask, normals from: https://github.com/nv-tlabs/FlexiCubes/blob/main/examples/render.py
# ! Depth from here: https://github.dev/NVlabs/diff-dope/blob/main/diffdope/diffdope.py
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr, rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else "all"
    )


def overlay_alpha(img, bg_img, alpha) -> torch.Tensor:
    # Blend foreground "img" over background "bg_img" using per-pixel alpha in [0,1]
    # torch.lerp(a, b, w) = a + w * (b - a)  => (1 - alpha) * bg + alpha * img
    return torch.lerp(bg_img, img, alpha)[0]


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(
        torch.clamp(dot(x, x), min=eps)
    )  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / length(x, eps)


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def auto_normals(self):
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        # specify dim argument to avoid deprecation warning in torch.cross
        nrm = safe_normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), eps=1e-20)
        self.nrm = nrm


def render_mesh(
    context: dr.RasterizeCudaContext | dr.RasterizeGLContext,
    mesh: Mesh,
    mv: torch.Tensor,
    mvp: torch.Tensor,
    resolution: list[int, int],
    return_types: list[str] = ["mask", "depth"],
    white_bg: bool = False,
    flip_vertical: bool = True,
    **kwargs,
) -> EasierDict:
    device = mesh.vertices.device
    bg = 255.0 if white_bg else 0.0
    bg_img = torch.fill(torch.zeros((*resolution, 3), device=device), bg)

    mesh_faces = mesh.faces.int()
    mesh_verts = mesh.vertices
    v_pos_clip = xfm_points(mesh_verts.unsqueeze(0), mvp)  # Rotate to camera coordinates
    mesh_faces = mesh_faces[:, [0, 2, 1]]
    rast, rast_db = dr.rasterize(context, v_pos_clip, mesh_faces, resolution)
    alpha = (rast[..., -1:] > 0).float()
    # assert alpha.sum() > 0, 'No mesh visible in the image'

    # Extract rotation (view) for transforming normals from object to camera/view space
    # if mv.ndim == 3:
    #     R_view = mv[0, :3, :3]
    # else:
    #     R_view = mv[:3, :3]

    out_dict = {}
    for ret_type in return_types:
        if ret_type == "mask":
            img = dr.antialias(alpha, rast, v_pos_clip, mesh_faces)
        elif ret_type == "normal":
            assert mesh.nrm is not None, "No normals provided"
            # Transform object-space vertex normals to view space (rigid -> transpose(R) == inverse)
            if mv.ndim == 3:
                R_view = mv[0, :3, :3]
            else:
                R_view = mv[:3, :3]
            nrm_view = safe_normalize(mesh.nrm @ R_view.T, eps=1e-20)
            normal_indices = (
                torch.arange(
                    0,
                    mesh.nrm.shape[0],
                    dtype=torch.int64,
                    device=mesh.vertices.device,
                )[:, None]
            ).repeat(1, 3)
            img, _ = interpolate(nrm_view.unsqueeze(0).contiguous(), rast, normal_indices.int())
        elif "vf" in ret_type:
            assert (features := getattr(mesh, ret_type)) is not None, "No vertex features provided"
            img, _ = interpolate(features.unsqueeze(0).contiguous(), rast, mesh_faces)
            img = overlay_alpha(img, bg_img, alpha)

        out_dict[ret_type] = img

    # * Sanitize the outputs to be aligned with OmniData
    ret = EasierDict(defaultdict(None))
    # mask, normals, depth = None, None, None
    if "mask" in return_types:
        mask = ((out_dict["mask"][0] + 1) / 2).clamp(0, 1)
        mask[mask < 0.5] = 0
        # with torch.no_grad():
        ret.mask = overlay_alpha(mask, bg_img[..., [0]], alpha).squeeze() * 255.0
        assert mask.squeeze().bool().sum() > 0, "No mesh visible in the image"
        mask_bool = mask.squeeze() > 0

    if "normal" in return_types:
        normals = out_dict["normal"][0]
        with torch.no_grad():
            normals *= torch.tensor([[1, -1, -1]], device=device)
            normals = (normals + 1) / 2
            normals = overlay_alpha(normals, bg_img, alpha)
        ret.normals = normals

    if "depth" in return_types:
        gb_pos, _ = interpolate(to_homog(mesh_verts), rast, mesh_faces, rast_db=rast_db)

        with torch.no_grad():
            shape_keep = gb_pos.shape
            gb_pos = gb_pos.reshape(shape_keep[0], -1, shape_keep[-1])[..., :3]

            depth = xfm_points(gb_pos.contiguous(), mv)
            depth = (depth.reshape(shape_keep)[..., 2] * -1).squeeze()
            depth[mask_bool] = standardize_tensor(depth[mask_bool]).clamp(0, 1)
            ret.depth = overlay_alpha(depth.unsqueeze(-1), bg_img[..., [0]], alpha).squeeze()

    if flip_vertical:
        if ret.get("mask") is not None:
            ret.mask = ret.mask.flip(0)
        if ret.get("normals") is not None:
            ret.normals = ret.normals.flip(0)
        if ret.get("depth") is not None:
            ret.depth = ret.depth.flip(0)

    return ret


class NVDiffRenderer(torch.nn.Module):
    FOCAL_LENGTH = 136.0

    def __init__(
        self,
        backend: str,
        z_near: float = 0.1,
        z_far: float = 1000.0,
        dtype=torch.float32,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.ctx = (
            dr.RasterizeCudaContext()
            if backend == "cuda" and torch.cuda.is_available()
            else dr.RasterizeGLContext()
        )
        self.register_buffer("z_near", torch.tensor(z_near, dtype=dtype, device=device))
        self.register_buffer("z_far", torch.tensor(z_far, dtype=dtype, device=device))

    def forward(
        self,
        v: torch.Tensor,
        f: torch.Tensor,
        camera: Camera,
        resolution: list[float, float],
        keys: list[str] = ["normal", "mask", "depth"],
        **kwargs,
    ) -> EasierDict:
        mv, mvp = camera.get_transform(
            res=resolution,
            device=self.device,
            z_near=self.z_near,
            z_far=self.z_far,
        )

        if v.shape[0] == 0:
            raise ValueError("Empty vertices")

        mesh = Mesh(v, f)
        for k, v in kwargs.items():
            if "vf" in k:
                setattr(mesh, k, v)

        mesh.auto_normals()
        render_buffers = render_mesh(
            self.ctx,
            mesh,
            mv.unsqueeze(0),
            mvp.unsqueeze(0),
            resolution,
            return_types=keys,
            center=camera.center,
            focal_x=camera.focal_length_x,
            focal_y=camera.focal_length_y,
            **kwargs,
        )
        return render_buffers
