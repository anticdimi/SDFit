import logging
import time
from functools import partial
from typing import Callable

import numpy as np
import plyfile
import skimage
import torch

from ...utils import EasierDict
from .flexicubes import FlexiCubes


def extract_mesh_flexicubes(
    sdf_func: Callable,
    res: int = 16,
    latent: torch.Tensor = None,
    x_nx3: torch.Tensor = None,
    device: str = 'cuda',
    grad_func: Callable = None,
    verbose: bool = False,
    grad: bool = False,
) -> EasierDict:
    fc = FlexiCubes(device)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(res)
    if grad:
        x_nx3.requires_grad = True

    x_nx3 = 1.5 * x_nx3  # add small margin to boundary
    inp = torch.hstack((latent.repeat(x_nx3.shape[0], 1), x_nx3)) if latent is not None else x_nx3
    sdf_vals = sdf_func(inp)
    if isinstance(sdf_vals, list):
        sdf_vals = sdf_vals[-1]
    if grad_func is not None:
        grad_func = partial(grad_func, outputs=sdf_vals)
    if verbose:
        print('Grid samples: ', x_nx3.min().item(), x_nx3.max().item())
        print('SDF values: ', sdf_vals.min().item(), sdf_vals.max().item())

    mesh_no_grad_v, mesh_no_grad_f, _ = fc(x_nx3, sdf_vals, cube_fx8, res, grad_func=grad_func)
    return EasierDict(v=mesh_no_grad_v, f=mesh_no_grad_f, samples=x_nx3, sdf_vals=sdf_vals)


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    return sdf


def create_mesh(decoder, latent_vec, filename, N=256, max_batch=32**3, offset=None, scale=None):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset).squeeze(1).detach().cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print('sampling takes: %f' % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + '.ply',
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[('vertex_indices', 'i4', (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, 'vertex')
    el_faces = plyfile.PlyElement.describe(faces_tuple, 'face')

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug('saving mesh to %s' % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        'converting to ply format and writing to file took {} s'.format(time.time() - start_time)
    )
