from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import EasierDict, calculate_centroid, normalize


def chamfer_distance(X1, X2):
    from external.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

    assert X1.shape[2] == 3
    Chamfer_3D = chamfer_3DDist().to(X1.device)
    dist_1, dist_2, idx_1, idx_2 = Chamfer_3D(X1, X2)
    return dist_1.sqrt(), dist_2.sqrt(), idx_1, idx_2


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        # if mask.size() != 3:
        #     mask = mask.unsqueeze(0)

        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


# ! Adapted from: https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class Loss(nn.Module):
    def __init__(self, weights_dict: dict):
        super(Loss, self).__init__()
        self.weights = weights_dict

    def _get_loss_msg(self, loss_dict: EasierDict, ignore_keys: list[str] = []) -> str:
        loss_msg = ""
        for key, loss_value in loss_dict.items():
            if key in ignore_keys:
                continue
            loss_msg += f"{key}={loss_value.item() * self.weights.get(key, 1.0):.4f}, "
        return loss_msg

    def forward(
        self,
        preds: torch.Tensor,
        tgts: torch.Tensor,
        return_msg: bool = False,
    ) -> EasierDict:
        loss = EasierDict()
        if "depth" in self.weights.keys():
            ssimae = ScaleAndShiftInvariantLoss()
            mask = tgts.mask.bool() | preds.mask.bool()
            loss["depth"] = ssimae(
                preds.depth[None, ...],
                tgts.depth[None, ...],
                mask[None, ...],
            )
        bpmask = preds.mask.bool()
        btmask = tgts.mask.bool()

        per_part_losses = defaultdict(list)
        for part_mask in tgts.cluster_masks:
            part_mask = part_mask.bool()
            real_part = part_mask & btmask
            weight = btmask.sum() / (real_part.sum() + 1e-20)

            # * Isolate part predictions and targets
            part_preds = EasierDict(
                {
                    k: v[real_part]
                    for k, v in preds.items()
                    if k in ["normals", "mask", "contour", "depth"]
                }
            )
            part_tgts = EasierDict(
                {
                    k: v[real_part]
                    for k, v in tgts.items()
                    if k in ["normals", "mask", "dist", "depth"]
                }
            )

            if "part_mask" in self.weights.keys():
                per_part_losses["part_mask"].append(
                    weight * F.mse_loss(part_preds.mask, part_tgts.mask)
                )
            if "part_depth" in self.weights.keys():
                part_pred = preds.depth * real_part
                part_tgt = tgts.depth * real_part
                per_part_losses["part_depth"].append(
                    weight
                    * ssimae(
                        part_pred[None, ...],
                        part_tgt[None, ...],
                        real_part[None, ...],
                    )
                )
            if "part_normals" in self.weights.keys():
                per_part_losses["part_normals"].append(
                    weight
                    * F.mse_loss(
                        normalize(part_preds.normals),
                        normalize(part_tgts.normals),
                    )
                )

        # * Mask MSE loss on the whole shape
        if "mask" in self.weights.keys():
            loss["mask"] = F.mse_loss(preds.mask, tgts.mask)
        # * Mask IoU loss on the whole shape
        if "mask_iou" in self.weights.keys():
            loss["mask_iou"] = 1 - ((bpmask & btmask).sum() / ((bpmask | btmask).sum() + 1e-20))
        if "l2_normals" in self.weights.keys():
            loss["l2_normals"] = F.mse_loss(
                normalize(preds.normals[bpmask]),
                normalize(tgts.normals[bpmask]),
            )
        # * Distance loss on the whole shape
        if "dist" in self.weights.keys():
            loss["dist"] = tgts.dist[preds.contour.bool()].mean()
        if "centroid" in self.weights.keys():
            bbox_coords = torch.nonzero(tgts.mask)
            min_coords = bbox_coords.min(dim=0)[0]
            max_coords = bbox_coords.max(dim=0)[0]
            target_mask_centroid = (min_coords + max_coords) / 2.0

            current_mask_centroid = calculate_centroid(preds.mask)
            loss["centroid"] = F.mse_loss(current_mask_centroid, target_mask_centroid)
        if "cd_canon_reg" in self.weights.keys():
            w_pred_v = preds.canon_shape_verts.unsqueeze(0)
            w_lookup_v = preds.canon_init_verts.unsqueeze(0)
            dist1, dist2, idx1, idx2 = chamfer_distance(w_lookup_v, w_pred_v)
            loss["cd_canon_reg"] = (dist1.mean() + dist2.mean()) / 2
        if "cycle_consistency" in self.weights.keys():
            w_pred_v = preds.canon_shape_verts.unsqueeze(0)
            w_lookup_v = preds.canon_init_verts.unsqueeze(0)
            _, _, _, idx2 = chamfer_distance(w_lookup_v, w_pred_v)
            # * For each hypothesis vertex, find the lookup one that is closest
            # * in canonical (warped) space
            closest_lookup = tgts.mesh_verts[idx2[0]]

            loss["cycle_consistency"] = F.mse_loss(preds.shape.v, closest_lookup)

        for k, v in per_part_losses.items():
            if k in self.weights.keys() and len(v) > 0:
                loss[k] = torch.stack(v).mean()

        for k, v in loss.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"Loss '{k}' is NaN! Something went wrong."

        loss["total_loss"] = sum(
            v * self.weights[k] for k, v in loss.items() if k in self.weights.keys()
        )

        if return_msg:
            loss["msg"] = self._get_loss_msg(loss)

        return loss
