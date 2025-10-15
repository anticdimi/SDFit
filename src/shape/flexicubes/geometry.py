import torch
from torch import nn

from .flexicubes import FlexiCubes


class FlexiCubesGeometry(nn.Module):
    def __init__(
        self,
        grid_res: int = 64,
        scale: float = 1.0,
        device: str = 'cuda',
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.grid_res = grid_res
        self.device = device

        self.fc = FlexiCubes(device)  # , weight_scale=0.5)
        self.verts, self.indices = self.fc.construct_voxel_grid(grid_res)
        self.verts += (torch.rand_like(self.verts, device=device) * 2 - 1) * 1e-4

        if not isinstance(scale, float):
            scale = self.verts.new_tensor(scale)
        self.verts = self.verts * scale

        all_edges = self.indices[:, self.fc.cube_edges].reshape(-1, 2)
        self.all_edges = torch.unique(all_edges, dim=0)

    def getAABB(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def get_mesh(
        self,
        x_nx3: torch.Tensor,
        sdf_n: torch.Tensor,
        weight_n: torch.Tensor = None,
        indices: torch.Tensor = None,
        is_training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if indices is None:
            indices = self.indices

        if weight_n is not None:
            beta_fx12 = weight_n[:, :12]
            alpha_fx8 = weight_n[:, 12:20]
            gamma_f = weight_n[:, 20]
        else:
            beta_fx12, alpha_fx8, gamma_f = None, None, None

        verts, faces, v_reg_loss = self.fc(
            x_nx3,
            sdf_n,
            indices,
            self.grid_res,
            beta_fx12=beta_fx12,
            alpha_fx8=alpha_fx8,
            gamma_f=gamma_f,
            training=is_training,
        )
        return verts, faces, v_reg_loss
