from pathlib import Path

import torch
from torch import nn

from ..utils import EasierDict
from .flexicubes import FlexiCubesGeometry
from .utils import normalize_points_to_unit_cube
from .workspace import load_decoder


class Shaper(nn.Module):
    def __init__(
        self,
        pretrained_exp_dir: str | Path,
        checkpoint: str,
        device: str | torch.device,
        mesh_res: int = 24,
        flexi_scale: list[float] = [1.8, 0.9, 0.9],
    ) -> None:
        super(Shaper, self).__init__()

        self.sdf_decoder = load_decoder(
            pretrained_exp_dir,
            checkpoint,
            device,
        )
        self.sdf_decoder.eval()

        self.geometry = FlexiCubesGeometry(
            mesh_res,
            flexi_scale,
            device,
        )

    @property
    def warper(self) -> nn.Module:
        return self.sdf_decoder.warper

    @property
    def template(self) -> nn.Module:
        return self.sdf_decoder.sdf_decoder

    def warp(self, x_nx3: torch.Tensor, latent: torch.Tensor, **kwargs) -> EasierDict:
        inp = torch.hstack((latent.repeat(x_nx3.shape[0], 1), x_nx3))
        return self.warper(inp, **kwargs)

    def forward_template(self, **kwargs) -> torch.Tensor:
        x_nx3 = self.geometry.verts
        sdf = self.sdf_decoder.forward_template(x_nx3)
        v, f, _ = self.geometry.get_mesh(
            x_nx3=x_nx3,
            sdf_n=sdf,
            is_training=False,
        )
        v = normalize_points_to_unit_cube(v)

        return EasierDict(v=v, f=f)

    def forward(self, latent: torch.Tensor, **kwargs) -> EasierDict:
        x_nx3 = self.geometry.verts
        inp = torch.hstack((latent.repeat(x_nx3.shape[0], 1), x_nx3))

        warped_xyzs = None
        sdf = self.sdf_decoder.forward(inp)
        if isinstance(sdf, tuple):
            warped_xyzs = sdf[0]
            sdf = sdf[1]

        v, f, l_dev = self.geometry.get_mesh(
            x_nx3=x_nx3,
            sdf_n=sdf,
            is_training=kwargs.get('is_training', False),
        )

        v = normalize_points_to_unit_cube(v)

        return EasierDict(
            v=v,
            f=f,
            warped_xyzs=warped_xyzs,
            l_dev=l_dev,
            edges=self.geometry.all_edges,
            sdf_vals=sdf,
        )
