#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import time

import numpy as np
import plyfile
import skimage.measure
import torch


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


def create_mesh_octree(
    decoder,
    latent_vec,
    filename,
    N=256,
    max_batch=32**3,
    offset=None,
    scale=None,
    clamp_func=None,
    volume_size=2.0,
):
    original_volume_size = 2.0
    start = time.time()
    ply_filename = filename

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-volume_size / 2.0, -volume_size / 2.0, -volume_size / 2.0]
    voxel_size = volume_size / (N - 1)

    overall_index = np.arange(0, N**3)
    samples = np.zeros([N**3, 4], dtype=np.float32)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index // N) % N
    samples[:, 0] = ((overall_index // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples = samples.reshape([N, N, N, 4])

    sdf_values = np.zeros([N, N, N], dtype=np.float32)
    dirty = np.ones([N, N, N], dtype=bool)
    grid_mask = np.zeros_like(dirty, dtype=bool)

    init_res = 64
    ignore_thres = volume_size / N / 4
    reso = N // init_res
    while reso > 0:
        grid_mask[0:N:reso, 0:N:reso, 0:N:reso] = True

        test_mask = np.logical_and(grid_mask, dirty)
        samples_ = samples[test_mask]
        samples_ = torch.from_numpy(samples_).cuda()
        sdf_ = []

        head = 0
        # print(samples_.shape[0])
        while head < samples_.shape[0]:
            query_idx = torch.arange(head, min(head + max_batch, samples_.shape[0])).long().cuda()
            s = decode_sdf(decoder, latent_vec, samples_[query_idx, :3]).view([-1]).detach()
            if clamp_func is not None:
                s = clamp_func(s)

            sdf_.append(s.cpu().numpy())
            head += max_batch

        sdf_values[test_mask] = np.concatenate(sdf_, axis=-1)

        if reso <= 1:
            break

        N_ds = N // reso - 1
        overall_index_ds = np.arange(0, N_ds**3)
        samples_ds = np.zeros([N_ds**3, 4], dtype=np.int32)

        # transform first 3 columns
        # to be the x, y, z index
        samples_ds[:, 2] = overall_index_ds % N_ds
        samples_ds[:, 1] = (overall_index_ds // N_ds) % N_ds
        samples_ds[:, 0] = ((overall_index_ds // N_ds) // N_ds) % N_ds
        samples_ds *= reso

        dirty_ds = dirty[
            samples_ds[:, 0] + reso // 2, samples_ds[:, 1] + reso // 2, samples_ds[:, 2] + reso // 2
        ]
        samples_ds = samples_ds[dirty_ds]
        v0 = sdf_values[samples_ds[:, 0], samples_ds[:, 1], samples_ds[:, 2]]
        v1 = sdf_values[samples_ds[:, 0], samples_ds[:, 1], samples_ds[:, 2] + reso]
        v2 = sdf_values[samples_ds[:, 0], samples_ds[:, 1] + reso, samples_ds[:, 2]]
        v3 = sdf_values[samples_ds[:, 0], samples_ds[:, 1] + reso, samples_ds[:, 2] + reso]
        v4 = sdf_values[samples_ds[:, 0] + reso, samples_ds[:, 1], samples_ds[:, 2]]
        v5 = sdf_values[samples_ds[:, 0] + reso, samples_ds[:, 1], samples_ds[:, 2] + reso]
        v6 = sdf_values[samples_ds[:, 0] + reso, samples_ds[:, 1] + reso, samples_ds[:, 2]]
        v7 = sdf_values[samples_ds[:, 0] + reso, samples_ds[:, 1] + reso, samples_ds[:, 2] + reso]

        vs = np.asarray([v0, v1, v2, v3, v4, v5, v6, v7])
        vmn = np.min(vs, axis=0)
        vmx = np.max(vs, axis=0)
        v_ = 0.5 * (vmx + vmn)
        clean_flag = (vmx - vmn) < ignore_thres
        for sample, v in zip(samples_ds[clean_flag], v_[clean_flag]):
            x, y, z = sample[0], sample[1], sample[2]
            sdf_values[x : x + reso, y : y + reso, z : z + reso] = v
            dirty[x : x + reso, y : y + reso, z : z + reso] = False

        reso //= 2

    end = time.time()
    logging.debug('sampling takes: %f' % (end - start))
    if '.ply' not in str(ply_filename):
        ply_filename += '.ply'

    # Calculate the scaling factor
    scaling_factor = original_volume_size / volume_size
    if scale is not None:
        scaling_factor = scaling_factor * scale

    scaling_factor = 1.0 / scaling_factor

    convert_sdf_samples_to_ply(
        sdf_values,
        voxel_origin,
        voxel_size,
        ply_filename,
        offset,
        scaling_factor,
    )


def convert_sdf_samples_to_ply(
    input_3d_sdf_array,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param input_3d_sdf_array: a float array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    if isinstance(input_3d_sdf_array, torch.Tensor):
        numpy_3d_sdf_tensor = input_3d_sdf_array.numpy()
    elif isinstance(input_3d_sdf_array, np.ndarray):
        numpy_3d_sdf_tensor = input_3d_sdf_array
    else:
        raise NotImplementedError

    verts, faces, normals, values = skimage.measure.marching_cubes(
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


def add_common_args(arg_parser):
    arg_parser.add_argument(
        '--debug',
        dest='debug',
        default=False,
        action='store_true',
        help='If set, debugging messages will be printed',
    )
    arg_parser.add_argument(
        '--quiet',
        '-q',
        dest='quiet',
        default=False,
        action='store_true',
        help='If set, only warnings will be printed',
    )
    arg_parser.add_argument(
        '--log',
        dest='logfile',
        default=None,
        help='If set, the log will be saved using the specified filename.',
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter('DeepSdf - %(levelname)s - %(message)s')
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
        sdf = decoder(inputs)[:, :1]
    else:
        try:
            latent_repeat = latent_vector.expand(num_samples, -1)
            inputs = torch.cat([latent_repeat, queries], 1)
            with torch.no_grad():
                sdf = decoder(inputs)[:, :1]
        except:
            raise RuntimeError('Failed to decode SDF')

    return sdf


def decode_warping(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    with torch.no_grad():
        warped, _ = decoder(inputs, output_warped_points=True)

    return warped
