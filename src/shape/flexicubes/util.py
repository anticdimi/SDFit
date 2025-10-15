# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import numpy as np
import torch

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(
        torch.clamp(dot(x, x), min=eps)
    )  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / length(x, eps)


def compute_vfov_from_focal_length(focal_length, image_height):
    return 2 * np.arctan(image_height / (2 * focal_length))


def compute_hfov_from_focal_length(focal_length, image_width):
    return 2 * np.arctan(image_width / (2 * focal_length))


def compute_focal_length_from_vfov(vfov, image_height):
    return image_height / (2 * np.tan(vfov / 2))


def compute_focal_length_from_hfov(hfov, image_width):
    return image_width / (2 * np.tan(hfov / 2))


def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = torch.tan(fovy / 2)
    return torch.tensor(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, 1 / -y, 0, 0],
            [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )


def translate(x, y, z, device=None):
    return torch.tensor(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=torch.float32, device=device
    )


@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def auto_normals(self):
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        nrm = safe_normalize(torch.cross(v1 - v0, v2 - v0), eps=1e-20)
        self.nrm = nrm


def load_mesh(path, device):
    import trimesh

    mesh_np = trimesh.load(path)
    vertices = torch.tensor(mesh_np.vertices, device=device, dtype=torch.float)
    faces = torch.tensor(mesh_np.faces, device=device, dtype=torch.long)

    # Normalize
    vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
    scale = 1.8 / torch.max(vmax - vmin).item()
    vertices = vertices - (vmax + vmin) / 2  # Center mesh on origin
    vertices = vertices * scale  # Rescale to [-0.9, 0.9]
    return Mesh(vertices, faces)


# def compute_sdf(points, vertices, faces):
#     face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices.clone().unsqueeze(0), faces)
#     distance = kaolin.metrics.trianglemesh.point_to_mesh_distance(
#         points.unsqueeze(0), face_vertices
#     )[0]
#     with torch.no_grad():
#         sign = (
#             kaolin.ops.mesh.check_sign(vertices.unsqueeze(0), faces, points.unsqueeze(0)) < 1
#         ).float() * 2 - 1
#     sdf = (sign * distance).squeeze(0)
#     return sdf


# def sample_random_points(n, mesh):
#     pts_random = (torch.rand((n // 2, 3), device='cuda') - 0.5) * 2
#     pts_surface = kaolin.ops.mesh.sample_points(mesh.vertices.unsqueeze(0), mesh.faces, 500)[
#         0
#     ].squeeze(0)
#     pts_surface += torch.randn_like(pts_surface) * 0.05
#     pts = torch.cat([pts_random, pts_surface])
#     return pts
