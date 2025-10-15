import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F

from ..utils import EasierDict, to_tensor


def compute_matching(
    set1: np.ndarray,
    set2: np.ndarray,
    n_components: int = -1,
    **kwargs: dict,
) -> np.ndarray:
    if n_components > 1:
        pca = PCA(n_components=n_components)
        embedded_data = pca.fit_transform(
            np.concatenate(
                (set1.cpu().numpy(), set2.cpu().numpy()),
                axis=0,
            )
        )
        print(
            f'Features explained variance: {pca.explained_variance_ratio_[:n_components].sum()},'
            + f'with {n_components} components'
        )
        embedded_data = (embedded_data - embedded_data.min()) / (
            embedded_data.max() - embedded_data.min()
        )

        embedded_data = to_tensor(embedded_data, dtype=torch.float32, device='cuda')
        transformed_set1 = embedded_data[: set1.shape[0]]
        transformed_set2 = embedded_data[set1.shape[0] :]
    else:
        transformed_set1 = to_tensor(set1, dtype=torch.float32, device='cuda')
        transformed_set2 = to_tensor(set2, dtype=torch.float32, device='cuda')

    distances_np = (
        1
        - torch.mm(
            F.normalize(transformed_set1, p=2, dim=1),
            F.normalize(transformed_set2, p=2, dim=1).T,
        )
        .cpu()
        .numpy()
    )

    col_indices = np.argmin(distances_np, axis=1)
    row_indices = np.arange(distances_np.shape[0])

    # Create the matching matrix
    matching_matrix = np.zeros((set1.shape[0], set2.shape[0]), dtype=np.int32)
    matching_matrix[row_indices, col_indices] = 1

    return matching_matrix


def solve_pnp_ransac(
    mesh_features: np.ndarray,
    img_features: np.ndarray,
    mesh_verts: np.ndarray,
    n_components: int,
    img_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    refine: bool = False,
    return_matching: bool = False,
    **kwargs: dict,
) -> EasierDict:
    device = mesh_verts.device

    matching_matrix = compute_matching(
        img_features,
        mesh_features,
        n_components=n_components,
    )
    max_indices = matching_matrix.argmax(axis=1).astype(np.int32)

    # Prepare 2D/3D correspondences
    points_3d = mesh_verts[max_indices].float()
    points_2d = img_points.float()

    # Camera parameters
    dist_coeffs = dist_coeffs if dist_coeffs is not None else torch.zeros((4, 1))
    cm_np = camera_matrix.float()
    dc_np = dist_coeffs.float()

    # PnP-RANSAC
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.numpy(),
        points_2d.numpy(),
        cm_np.numpy(),
        dc_np.numpy(),
        **kwargs.get('ransac_pnp'),
    )
    assert retval, 'PnP RANSAC failed'
    print('Inliers:', inliers.shape[0])
    foc_x, foc_y = cm_np[0, 0], cm_np[1, 1]

    # Optional refinement
    if refine:
        rvec, tvec = cv2.solvePnPRefineVVS(points_3d, points_2d, cm_np, dc_np, rvec, tvec)
        foc_x = cm_np[0, 0]
        foc_y = cm_np[1, 1]

    Rs, ts = [], []
    # Align axes from OpenCV (x right, y down, z forward)
    # to OpenGL/NVDiff (x right, y up, z backward):
    R_cv = cv2.Rodrigues(rvec)[0]
    t_cv = tvec.squeeze()
    S = np.diag([1.0, -1.0, -1.0])
    R_gl_w2c = S @ R_cv  # world -> camera (OpenGL axes)
    t_gl_w2c = S @ t_cv

    R_gl_c2w = R_gl_w2c.T
    t_gl_c2w = -R_gl_w2c.T @ t_gl_w2c

    # Primary candidate (camera pose, OpenGL axes)
    Rs.append(R_gl_c2w.copy())
    ts.append(t_gl_c2w.copy())

    # Secondary candidate: flip XZ plane w.r.t. the first pose to address the
    # left-right ambiguity in matching.
    flip_x = np.diag([-1.0, 1.0, -1.0])
    R2_w2c = flip_x @ R_gl_w2c
    R2_c2w = R2_w2c.T
    t2_c2w = -R2_w2c.T @ t_gl_w2c
    Rs.append(R2_c2w.copy())
    ts.append(t2_c2w.copy())

    return EasierDict(
        ext_Rs=[to_tensor(rot, device=device) for rot in Rs],
        ext_ts=[to_tensor(t, device=device) for t in ts],
        focal_x=to_tensor(foc_x, device=device),
        focal_y=to_tensor(foc_y, device=device),
        inliers=inliers,
        verts_idx=to_tensor(max_indices, device=device),
        matching=to_tensor(matching_matrix, device=device) if return_matching else None,
    )
