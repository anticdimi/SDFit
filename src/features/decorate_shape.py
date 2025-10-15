import random
from time import time

import numpy as np
import torch
from PIL import Image
from pytorch3d.ops import ball_query
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from ..utils import EasierDict, arange_pixels, normals_from_depth_sobel
from .extractor_controlnet import add_texture_to_render
from .extractor_dino import get_dino_features
from .render import batch_render

VERTEX_GPU_LIMIT = 35000


@torch.no_grad()
def fuse_unet_features(feat_dict, mode='bilinear'):
    ft_2 = feat_dict['unet_fmap_2']
    ft_4 = feat_dict['unet_fmap_4']
    features_gather_2_4 = torch.cat(
        [ft_2, F.interpolate(ft_4, size=(ft_2.shape[-2:]), mode=mode)], dim=1
    )
    return features_gather_2_4


@torch.no_grad()
def extract_controlnet_dino_features(
    input_img: np.ndarray,
    normals_img: np.ndarray,
    depth_img: np.ndarray,
    mask_indices: torch.Tensor,
    dino_model: nn.Module,
    controlnet_model: nn.Module,
    prompt: str,
    device: str,
    n_steps: int = 100,
    fuse_unet: bool = True,
    control: list = ['depth', 'normal'],
    return_tex_image: bool = False,
    return_separate: bool = False,
    dino_from_original: bool = False,
    alpha: float = 0.5,
    extr_mode: str = 'bilinear',
    **kwargs,
) -> EasierDict:
    # ! 1. ControlNet UNet features
    diffusion_output = add_texture_to_render(
        controlnet_model,
        input_img.astype(np.uint8),
        depth_img,
        prompt,
        normals_img if 'normal' in control else None,
        return_image=True,
        num_inference_steps=n_steps,
        **kwargs,
    )
    H, W = input_img.shape[:2]
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    features_unet = (
        fuse_unet_features(diffusion_output[0], mode=extr_mode)
        if fuse_unet
        else diffusion_output[0]['unet_fmap_2']
    )

    ft = torch.nn.Upsample(size=(H, W), mode=extr_mode)(features_unet).float().to(device)
    ft_dim = ft.size(1)
    aligned_unet = F.grid_sample(ft, grid.float(), mode=extr_mode, align_corners=False).reshape(
        1, ft_dim, -1
    )
    aligned_unet = F.normalize(aligned_unet, dim=-1)

    if dino_from_original:
        orig_dino_features = get_dino_features(
            device, dino_model, Image.fromarray(input_img.astype(np.uint8))
        )
        orig_dino_features = F.grid_sample(
            orig_dino_features, grid, mode=extr_mode, align_corners=False
        ).reshape(1, 768, -1)
        aligned_dino_features = F.normalize(orig_dino_features, dim=-1)
    else:
        # ! 2. DINOv2 on ControlNet-textured image
        aligned_dino_features = get_dino_features(device, dino_model, diffusion_output[1][0])
        aligned_dino_features = F.grid_sample(
            aligned_dino_features, grid, mode=extr_mode, align_corners=False
        ).reshape(1, 768, -1)
        aligned_dino_features = F.normalize(aligned_dino_features, dim=-1)

    # ! 3. Fuse features
    aligned_fused_features = torch.hstack(
        [aligned_unet * alpha, aligned_dino_features * (1 - alpha)]
    )
    features_per_pixel = aligned_fused_features[0, :, mask_indices.cpu()]

    ret_dict = {}
    ret_dict['fused_features'] = features_per_pixel.permute(1, 0)
    if return_separate:
        ret_dict['controlnet_features'] = aligned_unet[0, :, mask_indices.cpu()].permute(1, 0)
        ret_dict['dino_features_tex'] = aligned_dino_features[0, :, mask_indices.cpu()].permute(
            1, 0
        )
    else:
        ret_dict['fused_features'] = features_per_pixel.permute(1, 0)

    if return_tex_image:
        ret_dict['tex_image'] = diffusion_output[1][0]

    return EasierDict(ret_dict)


def decorate_shape_controlnet(
    device,
    pipe,
    dino_model,
    mesh,
    prompt,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_normal_map=True,
    mesh_vertices=None,
    bq=True,
    prompts_list=None,
    fuse_unet=True,
    num_inference_steps=30,
    debug=False,
):
    feature_dims = 1600 + 768 if fuse_unet else 1280 + 768  # * unet decoder L2 & L4 + dino
    t1 = time()

    mesh_vertices = mesh.verts_list()[0]
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()  # .cpu()
    ball_drop_radius = maximal_distance * tolerance

    batched_renderings, normal_batched_renderings, camera, depth = batch_render(
        device,
        mesh,
        num_views,
        H,
        W,
        use_normal_map,
    )
    if debug:
        print('Rendering complete')
    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()
    batched_renderings = batched_renderings.cpu()
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    camera = camera.cpu()
    normal_map_input = None
    depth = depth.cpu()

    torch.cuda.empty_cache()
    ft_per_vertex = torch.zeros((len(mesh_vertices), feature_dims)).half()  # .to(device)
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()  # .to(device)
    for idx in tqdm(range(len(batched_renderings)), desc='Extracting features from renders'):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]
        world_coords = (
            camera[idx].unproject_points(xy_depth, world_coordinates=True, from_ndc=True)  # .cpu()
        ).to(device)

        depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)
        if prompts_list is not None:
            prompt = random.choice(prompts_list)

        diffusion_input_img = (batched_renderings[idx, :, :, :3].cpu().numpy() * 255).astype(
            np.uint8
        )

        _depth = depth_map.cpu().numpy().transpose(1, 2, 0)
        mask = _depth != -1
        _depth = (_depth.max() - _depth) * mask
        normal_map_input = (
            normals_from_depth_sobel(
                _depth.squeeze() * 255.0,
                bg_mask=~mask,
                ksize=9,
                smoothing=False,
                expand_mask=True,
            )
            * mask
        )

        features = extract_controlnet_dino_features(
            input_img=diffusion_input_img,
            normals_img=normal_map_input,
            depth_img=depth_map,
            mask_indices=indices,
            dino_model=dino_model,
            controlnet_model=pipe,
            prompt=prompt,
            device=device,
            n_steps=num_inference_steps,
            fuse_unet=fuse_unet,
            return_tex_image=True if debug else False,
        )
        features_per_pixel = features.fused_features.permute(1, 0).cpu()

        if bq:
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    mesh_vertices.unsqueeze(0),
                    K=10,
                    radius=ball_drop_radius,
                    return_nn=False,
                )
                .idx[0]
                .cpu()
            )
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            ft_per_vertex_count[queried_indices[mask]] += 1
            ft_per_vertex[queried_indices[mask]] += features_per_pixel.repeat_interleave(
                repeat, dim=1
            ).T
        else:
            distances = torch.cdist(world_coords, mesh_vertices, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += features_per_pixel.T
            ft_per_vertex_count[closest_vertex_indices] += 1

    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
    if debug:
        print('Number of missing features: ', missing_features)
        print('Copied features from nearest vertices')

    if missing_features > 0:
        if debug:
            print('Filling missing features based on nearest neighbors')
        filled_indices = ft_per_vertex_count[:, 0] != 0
        missing_indices = ft_per_vertex_count[:, 0] == 0
        distances = torch.cdist(mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2)
        closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][closest_vertex_indices, :]
    t2 = time() - t1
    t2 = t2 / 60
    if debug:
        print('Time taken in mins: ', t2)
    return ft_per_vertex
