import warnings
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import torch
import transformers
from pytorch3d.io import IO
from pytorch3d.structures import Meshes

from ..features import (
    decorate_shape_controlnet,
    extract_controlnet_dino_features,
    init_controlnet,
    init_dino,
)
from ..utils import mkdir, normals_from_depth_sobel

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


def do_lookup(
    img: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    shape_collection_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(shape_collection_path, str):
        shape_collection_path = Path(shape_collection_path)
    assert shape_collection_path.exists(), f'Path {shape_collection_path} does not exist'
    assert img.shape[:2] == mask.shape[:2], 'Input image and mask must have the same shape'

    masked_img = (img * mask.astype(bool)[..., None] / 255.0) * 2 - 1
    clip_model, clip_prep = (
        transformers.CLIPModel.from_pretrained(
            'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
            offload_state_dict=True,
        ),
        transformers.CLIPProcessor.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'),
    )
    if torch.cuda.is_available():
        clip_model.cuda()

    inputs = clip_prep(images=masked_img, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values']).float()
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    with h5py.File(shape_collection_path, 'r') as collection_file:
        uids, feats = [], []
        for k in collection_file.keys():
            uids.append(k)
            grp = collection_file[k]
            if 'openshape_emb' in grp:
                feats.append(grp['openshape_emb'][()])

        shape_feats = torch.tensor(np.stack(feats)).to(DEVICE)
        shape_feats = shape_feats / shape_feats.norm(p=2, dim=-1, keepdim=True)

        sims = (image_features @ shape_feats.T).squeeze(0)
        best_uid = uids[sims.argmax().item()]

        v, f = collection_file[best_uid]['vertices'][()], collection_file[best_uid]['faces'][()]
        latent = collection_file[best_uid]['latent'][()]

    print(f'Best matching shape UID: {best_uid} with similarity {sims.max().item():.4f}')

    return v, f, latent


def decorate_2d(
    inp_rgb: torch.Tensor,
    inp_mask: torch.Tensor,
    inp_depth: torch.Tensor,
    text_prompt: str,
    num_inference_steps: int = 30,
):
    dinov2 = init_dino(DEVICE)
    controlnet = init_controlnet(DEVICE, ['depth', 'normal'])

    inp_mask = inp_mask.bool().unsqueeze(-1)
    mask_indices = inp_mask.view(-1)

    inp_normals = normals_from_depth_sobel(
        inp_depth.cpu().numpy(),
        bg_mask=inp_mask.squeeze().cpu().numpy(),
    )

    # * Mask out background
    inp_depth = (inp_depth.unsqueeze(-1) * inp_mask).squeeze()
    inp_img = inp_rgb * inp_mask

    dec_output = extract_controlnet_dino_features(
        input_img=inp_img.cpu().numpy(),
        normals_img=inp_normals,
        depth_img=inp_depth.cpu().numpy(),
        mask_indices=mask_indices,
        dino_model=dinov2,
        controlnet_model=controlnet,
        prompt=text_prompt,
        n_steps=num_inference_steps,
        device=DEVICE,
        fuse_unet=True,
        use_normal_map=True,
    )

    return dec_output['fused_features']


def decorate_3d(
    mesh: Path | Meshes | str,
    text_prompt: str,
    resolution: int = 224,
    n_views: int = 18,
    num_inference_steps: int = 30,
    debug: bool = False,
):
    dinov2 = init_dino(DEVICE)
    controlnet = init_controlnet(DEVICE, ['depth', 'normal'])

    if isinstance(mesh, (Path, str)):
        mesh = IO().load_mesh(mesh, include_textures=False)
    else:
        assert isinstance(mesh, Meshes), 'Mesh should be a Path, str, or Pytorch3D Meshes object'

    features = decorate_shape_controlnet(
        device=DEVICE,
        pipe=controlnet,
        num_inference_steps=num_inference_steps,
        dino_model=dinov2,
        mesh=mesh,
        prompt=text_prompt,
        num_views=n_views,
        H=resolution,
        W=resolution,
        bq=True,
        prompts_list=None,
        fuse_unet=True,
        use_normal_map=True,
        debug=debug,
    )

    return features


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--img_path', type=Path)
    parser.add_argument('--mask_path', type=Path)
    parser.add_argument('--depth_path', type=Path)
    parser.add_argument('--prompt', type=str, default='A picture model of a \{CATEGORY\}')
    parser.add_argument('--out_dir', type=Path, default='./outputs')
    parser.add_argument('--n_steps', type=int, default=30, help='Number of steps for diffusion')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    features = decorate_2d(
        inp_rgb=args.img_path,
        inp_mask=args.mask_path,
        inp_depth=args.depth_path,
        text_prompt=args.prompt,
        num_inference_steps=args.n_steps,
    )

    img_name = args.img_path.stem

    save_path = mkdir(args.out_dir) / f'{img_name}.npz'
    print(f'Saving features to {save_path}')
    np.savez_compressed(save_path, feat=features.cpu().numpy())
