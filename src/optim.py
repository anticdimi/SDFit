import os
import time
from collections import defaultdict
from math import isclose
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import tensorboardX as tbx
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm
from videoio import VideoWriter

from .diffrend import Camera
from .utils import (
    EasierDict,
    apply_rigid_transform,
    find_files,
    human_readable_time,
    mkdir,
    normalize,
    rot_6d_to_matrix,
    save_image,
    standardize,
)


def init_optimizer(parameters: EasierDict, vars_cfg: EasierDict) -> tuple:
    def lambda_function(it, lr_init, lr_min, T_max):
        if lr_init == lr_min:
            return lr_min
        return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * it / T_max))

    parameter_groups, lambdas = [], []
    for var_name, var_value in parameters.items():
        parameter_groups.append({'name': var_name, 'params': var_value, 'lr': 1})
        parameter_groups[-1]['lr'] = 1  # ! Hack to work with my lambda function
        vcfg = vars_cfg[var_name]
        lr_init = vcfg['lr']
        lr_min = vcfg.get('lr_min', lr_init)
        T_max = vcfg['T_max']

        lambdas.append(
            lambda epoch, lr_init=lr_init, lr_min=lr_min, T_max=T_max: lambda_function(
                epoch, lr_init, lr_min, T_max
            )
        )
    optimizer = torch.optim.Adam(parameter_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    return optimizer, scheduler


class FittingMonitor(object):
    def __init__(self, config: DictConfig, save_dir: Path, max_iters: int, **kwargs) -> None:
        super(FittingMonitor, self).__init__()
        self.max_iters = max_iters

        self.stage_name = kwargs.get('stage_name', 'Fitting')

        self.save_dir = save_dir
        self.imgs_dir = mkdir(save_dir / 'imgs')
        self.params_dir = mkdir(save_dir / 'params')

        self.render_every = config.log.render_every
        self.print_every = config.log.print_every
        self.params_every = config.log.params_every

        self.vwriter = VideoWriter(
            self.save_dir / 'overlay.mp4',
            resolution=(224, 224),
        )
        self.dvwriter = (
            VideoWriter(
                self.save_dir / 'debug.mp4',
                resolution=(4 * 224, 2 * 224),
            )
            if kwargs.get('log_debug', True)
            else None
        )
        self.tb_logger = tbx.SummaryWriter(self.save_dir / 'summary')

        self.device = config.device
        self.cfg = config

        self.min_loss = torch.inf

    def __enter__(self) -> 'FittingMonitor':
        self.steps = 0
        self.start_time = time.time()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        end_time = time.time()
        print(f'\tTime: {human_readable_time(end_time - self.start_time)}')
        # * Sanitize
        self.vwriter.close()
        if self.dvwriter is not None:
            self.dvwriter.close()

        if self.imgs_dir.exists() and not any(self.imgs_dir.glob('*.png')):
            self.imgs_dir.rmdir()
        if self.params_dir.exists() and not any(self.params_dir.glob('*.npz')):
            self.params_dir.rmdir()

    # *################################ Hooks ##########################################
    def _log_image(self, image: torch.Tensor | np.ndarray, tag: str) -> None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 3:
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
        save_image(image, self.imgs_dir / f'{tag}_{self.steps:04d}.png')

    def _log_vframe(self, image: torch.Tensor | np.ndarray) -> None:
        self.vwriter.write(image)

    @torch.no_grad()
    def _render_hook(self, pred_dict: EasierDict, tgt_dict: EasierDict) -> None:
        rendered = defaultdict()
        if 'mask' in pred_dict.keys():
            obs_mask = pred_dict.mask.detach().cpu().numpy().astype('uint8')
            tgt_mask = tgt_dict.mask.detach().cpu().numpy().astype('uint8')
            mask = np.vstack((tgt_mask, obs_mask))
            rendered['mask'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if 'depth' in pred_dict.keys():
            obs_depth = standardize(pred_dict.depth.detach()).cpu().numpy()
            tgt_depth = standardize(tgt_dict.depth.detach()).cpu().numpy()

            def viridis(tensor):
                import matplotlib.pyplot as plt

                # Apply viridis colormap
                viridis_array = plt.cm.viridis(tensor)
                viridis_tensor = (viridis_array[:, :, :3] * 255).astype(np.uint8)
                return viridis_tensor

            rendered['depth'] = np.vstack((viridis(tgt_depth), viridis(obs_depth)))

        if 'contour' in pred_dict.keys():
            tgt_dist = tgt_dict.dist.detach().cpu().numpy() * 255.0
            obs_contour = pred_dict.contour.detach().cpu().numpy() * 255.0
            dist = np.vstack((tgt_dist, obs_contour)).astype('uint8')
            rendered['dist'] = cv2.cvtColor(dist, cv2.COLOR_GRAY2RGB)

        if 'normals' in pred_dict.keys():
            tgt_normals = normalize(tgt_dict.normals.detach()).cpu().numpy() * 255.0
            obs_normal = normalize(pred_dict.normals.detach()).cpu().numpy() * 255.0
            # * Mask predicted normals on top of the input image
            overlay = tgt_dict.input_img.float().cpu().numpy()
            overlay[obs_mask > 0] = obs_normal[obs_mask > 0]
            rendered['normal'] = np.vstack((tgt_normals, overlay))

            # save_image(overlay, self.imgs_dir / f'{self.steps:04d}_overlay.png')
            self._log_vframe(overlay.astype(np.uint8))

        if self.steps == self.max_iters - 1:
            save_image(overlay, self.save_dir / f'final_{self.stage_name}.png')
        if self.dvwriter is not None:
            self.dvwriter.write(np.hstack(list(rendered.values())).astype(np.uint8))

    def _log_hook(
        self,
        params: EasierDict,
        buffers: EasierDict,
        tgts: EasierDict,
        loss_dict: EasierDict,
        **kwargs,
    ) -> None:
        if self.steps == 0 or (self.render_every > 0 and self.steps % self.render_every == 0):
            self._render_hook(buffers, tgts)

        if self.print_every > 0 and self.steps % self.print_every == 0:
            print(loss_dict.msg)
        if self.params_every > 0 and self.steps % self.params_every == 0:
            np.savez(self.params_dir / f'{self.steps:04d}_params.npz', **params.cpu().to_dict())
        if loss_dict.total_loss < self.min_loss:
            self.min_loss = loss_dict.total_loss
            prev_best = find_files(self.params_dir.parent.parent, 'Shape_Fitting_best_*.npz')
            if len(prev_best) > 0:
                os.unlink(prev_best[0])
            np.savez(
                self.params_dir.parent.parent / f'Shape_Fitting_best_{self.steps:04d}.npz',
                **params.cpu().to_dict(),
            )

        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                self.tb_logger.add_scalar(f'stage_{self.stage_name}/{k}', v.item(), self.steps)

    #!################################ <\end> Hooks ##########################################

    def run_fitting(
        self,
        optimizer: torch.optim.Optimizer,
        closure: Callable[..., Any],
        scheduler: nn.Module,
        desc: str = None,
        no_change_patience: int = 20,
        no_change_tol: float = 1e-4,
    ) -> list[float]:
        losses = []
        # Track loss stability
        prev_loss_val: float | None = None
        stable_count = 0
        for _ in (
            pbar := tqdm(
                range(self.max_iters),
                total=self.max_iters,
                desc=self.stage_name if desc is None else desc,
            )
        ):
            loss = optimizer.step(closure)
            if scheduler is not None:
                scheduler.step()
            current_val = loss.total_loss.item()
            pbar.set_postfix_str(f'Loss: {current_val:.4f}')
            losses.append(loss)

            # Early stop if loss hasn't changed for `no_change_patience` iters
            if prev_loss_val is not None:
                if isclose(
                    current_val, prev_loss_val, rel_tol=no_change_tol, abs_tol=no_change_tol
                ):
                    stable_count += 1
                else:
                    stable_count = 0
            prev_loss_val = current_val

            if no_change_patience > 0 and stable_count >= no_change_patience:
                pbar.write(
                    f'Early stopping: loss unchanged for {no_change_patience} iterations (tol={no_change_tol}).'
                )
                break

        return losses

    # *################################ Closure ##########################################
    def create_fitting_closure(
        self,
        optimizer: torch.optim.Optimizer,
        shape_prior: nn.Module,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        scale: torch.Tensor,
        latent: torch.Tensor,
        targets: EasierDict,
        renderer: nn.Module | Callable[..., Any],
        camera: Camera,
        resolution: list[float, float],
        loss_fn: nn.Module | Callable[..., Any],
        mesh_v: torch.Tensor | None = None,
        mesh_f: torch.Tensor | None = None,
        contour_finder: nn.Module = None,
        is_training: bool = False,
        # save_dir,
    ) -> Callable[..., Any]:
        def fitting_func(backward=True) -> torch.Tensor:
            if backward:
                optimizer.zero_grad()
            if mesh_v is None:
                shape = shape_prior(latent, is_training=is_training)
            else:
                shape = EasierDict(v=mesh_v, f=mesh_f)

            rotation_matrix = rot_6d_to_matrix(rotation)
            posed_v = apply_rigid_transform(
                points=shape.v,
                rotation_matrix=rotation_matrix,
                translation=translation,
                scale=scale,
            )

            pred = renderer(
                posed_v,
                shape.f,
                camera,
                resolution,
            )
            pred.latent = latent
            pred.shape = shape

            if latent is not None:
                pred.canon_init_verts = shape_prior.warp(targets.mesh_verts, targets.init_latent)[0]
                pred.canon_shape_verts = shape_prior.warp(shape.v, latent, step=8.0)[0]

            if contour_finder:
                pred.contour = contour_finder(pred.mask.unsqueeze(0)).squeeze()

            loss_dict = loss_fn(pred, targets, return_msg=True)

            if backward:
                loss_dict.total_loss.backward()

            current_state = EasierDict(
                rotation=rotation_matrix.detach(),
                translation=translation.detach(),
                scale=scale.detach(),
                latent=latent.detach() if latent is not None else None,
                ext_R=camera.rotation.detach(),
                ext_t=camera.translation.detach(),
                focal_x=camera.focal_length_x.detach(),
                focal_y=camera.focal_length_y.detach(),
                loss=loss_dict.detach(),
                v=shape.v.detach().unsqueeze(0),
                f=shape.f.detach().unsqueeze(0),
            )
            self._log_hook(
                current_state,
                pred,
                targets,
                loss_dict,
                param_groups=optimizer.param_groups,
            )
            self.steps += 1

            return loss_dict

        return fitting_func
