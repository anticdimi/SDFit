import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

warnings.filterwarnings("ignore", category=UserWarning)
from src.diffrend import Camera, NVDiffRenderer
from src.init import decorate_2d, decorate_3d, do_lookup, solve_pnp_ransac
from src.loss import Loss
from src.optim import FittingMonitor, init_optimizer
from src.shape import Shaper, create_mesh_octree
from src.utils import (
    EasierDict,
    EdgeDetector,
    dump_yaml,
    fix_seeds,
    human_readable_time,
    load_config,
    load_image,
    load_mesh,
    matrix_to_rot6d,
    mkdir,
    normalize_depth,
    to_tensor,
)

SEPARATOR = "\n" + "=" * 50 + "\n"


def load_data(data_config: DictConfig, device: str | torch.device) -> EasierDict:
    inp_data = EasierDict()
    input_root = Path(data_config.input_path)

    # * Load RGB image (not used for fitting)
    inp_data["input_img"] = to_tensor(load_image(input_root / "rgb.png"))[..., :3]

    # * Load mask image (used for fitting)
    inp_data["mask"] = to_tensor(load_image(input_root / "mask.png", mode="L"), dtype=torch.float32)

    # * Load normals image (used for fitting)
    inp_data["normals"] = to_tensor(load_image(input_root / "normals.png")).float()
    inp_data["normals"][inp_data["mask"] == 0] = 0

    # * Load depth image (used for fitting)
    depth = to_tensor(load_image(input_root / "depth.png", mode="L"), dtype=torch.float32)
    valid_pixels = inp_data["mask"] > 0
    norm_depth = normalize_depth(depth[valid_pixels])
    inp_data["depth"] = torch.zeros_like(depth)
    inp_data["depth"][valid_pixels] = 1 - norm_depth

    # * Load bdist image (used for fitting)
    inp_data["dist"] = torch.tensor(load_image(input_root / "bdist.png", mode="L") / 255.0).float()

    # * Load estimated camera params
    pred_cam_params = dict(np.load(input_root / "cam_params.npz", allow_pickle=True))
    orig_shape = pred_cam_params["shape"]  # expected (H, W)
    H, W = int(orig_shape[0]), int(orig_shape[1])
    pred_cam_int = torch.zeros(3, 3)
    # Center in pixel coords: (cx, cy) = (W/2, H/2)
    cx, cy = W / 2.0, H / 2.0
    foc = to_tensor(pred_cam_params["focal"], dtype=torch.float32)
    center = to_tensor([cx, cy], dtype=torch.float32)
    pred_cam_int[:-1, -1] = center
    K = pred_cam_int.float()
    K[0, 0] = foc
    K[1, 1] = foc
    K[2, 2] = 1.0
    inp_data["cam_params"] = EasierDict(K=K, focal=foc, center=center)

    # * Load image and shape features
    inp_data["img_features"] = (
        None
        if not (input_root / "img_features.npz").exists()
        else to_tensor(np.load(input_root / "img_features.npz")["feat"])
    )

    inp_data["shape_features"] = (
        None
        if not (input_root / "shape_features.npz").exists()
        else to_tensor(np.load(input_root / "shape_features.npz")["feat"])
    )

    if (input_root / "init_lookup.ply").exists():
        v, f = load_mesh(str(input_root / "init_lookup.ply"))
        inp_data["mesh_verts"] = to_tensor(v, dtype=torch.float32)
        inp_data["mesh_faces"] = to_tensor(f, dtype=torch.long)
        inp_data["init_latent"] = to_tensor(
            np.load(input_root / "init_latent.npz")["latent"][None, ...],
            dtype=torch.float32,
        )
    else:
        inp_data["mesh_verts"] = None
        inp_data["mesh_faces"] = None
        inp_data["init_latent"] = None

    return inp_data


def main(opt: DictConfig):
    save_root = mkdir(Path(opt.save_root))
    print(f"Saving logs to {save_root.resolve()}")
    dump_yaml(save_root / "config.yaml", opt)

    # * Data loading
    inp_data = load_data(opt.data, device=opt.device)
    resolution = inp_data.mask.shape[:2]

    print("SDFit running on: " + "CUDA" if torch.cuda.is_available() else "CPU")
    print(f"\tTargets path: {opt.data.input_path}", end=SEPARATOR)

    ############################################################
    # ! Step 0: LookUp + Decorate
    if inp_data.mesh_verts is None:
        ckpt_start = time.time()
        print("[Step 0.0] Shape LookUp.")
        v, f, init_latent = do_lookup(
            img=inp_data.input_img.cpu().numpy(),
            mask=inp_data.mask.cpu().numpy(),
            shape_collection_path=opt.data.shape_collection_path,
        )
        inp_data["mesh_verts"] = to_tensor(v, dtype=torch.float32)
        inp_data["mesh_faces"] = to_tensor(f, dtype=torch.long)
        inp_data["init_latent"] = to_tensor(init_latent[None, ...], dtype=torch.float32)
        print(
            f"\tTime (with ckpt loading): {human_readable_time(time.time() - ckpt_start)}",
            end=SEPARATOR,
        )
    else:
        assert inp_data.init_latent is not None
        assert inp_data.mesh_faces is not None
        print("[Step 0.0] Init shape provided.", end=SEPARATOR)
        v, f, init_latent = (
            inp_data.mesh_verts.cpu().numpy(),
            inp_data.mesh_faces.cpu().numpy(),
            inp_data.init_latent.cpu().numpy()[0],
        )

    # * Decorate with 3D ControlNet + DINOv2
    if inp_data.shape_features is None:
        ckpt_start = time.time()
        print("[Step 0.1] Decorating init shape with ControlNet + DINOv2 features.")
        inp_data.mesh_faces = inp_data.mesh_faces[..., [2, 1, 0]]  # flip face orientation
        inp_data["shape_features"] = decorate_3d(
            mesh=Meshes(
                verts=[inp_data.mesh_verts],
                faces=[inp_data.mesh_faces],
                textures=TexturesVertex([torch.full_like(inp_data.mesh_verts, 0.7)]),
            ).to(opt.device),
            text_prompt=f"A picture of {opt.shape_prior.category}, photorealistic; high quality, detailed;",
            n_views=8,
            num_inference_steps=50,
        ).to(torch.float32)
        print(f"\tTime (with ckpt loading): {human_readable_time(time.time() - ckpt_start)}")
    else:
        print("[Step 0.1] Init shape decorated.")

    if inp_data.img_features is None:
        ckpt_start = time.time()
        print("[Step 0.2] Decorating input image with ControlNet + DINOv2 features.")
        inp_data["img_features"] = decorate_2d(
            inp_rgb=inp_data.input_img,
            inp_mask=inp_data.mask,
            inp_depth=inp_data.depth,
            text_prompt=f"A picture of {opt.shape_prior.category}, photorealistic; high quality, detailed;",
            num_inference_steps=30,
        ).to(torch.float32)
        print(
            f"\tTime (with ckpt loading): {human_readable_time(time.time() - ckpt_start)}",
            end=SEPARATOR,
        )

    else:
        print("[Step 0.2] Input image decorated.", end=SEPARATOR)

    if opt.shape_stage.loss_cfg.get("part_based", False):
        from sklearn.cluster import KMeans

        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = torch.from_numpy(
            kmeans.fit_predict(inp_data["img_features"].cpu().numpy())
        ).to(torch.long)
        mask_bool = inp_data["mask"].bool()
        pix_aligned_lbls = torch.full_like(inp_data["mask"], -1, dtype=torch.long)
        pix_aligned_lbls[mask_bool] = cluster_labels
        cluster_masks = torch.zeros((n_clusters, *inp_data["mask"].shape), dtype=torch.float32)
        cluster_masks[:, mask_bool] = F.one_hot(cluster_labels, n_clusters).T.float()
        inp_data["cluster_masks"] = cluster_masks

    ############################################################

    init_values = EasierDict()

    ############################################################
    # ! Step 1.1: Object/Camera Ransac-PnP
    ext_R = torch.eye(3, dtype=torch.float32)
    ext_t = torch.tensor([0, 0, 0], dtype=torch.float32)
    focal_x = inp_data.cam_params.K[0, 0]
    focal_y = inp_data.cam_params.K[1, 1]

    ckpt_start = time.time()

    img_points = torch.nonzero(inp_data.mask)
    img_points = img_points[:, [1, 0]]  # ->  (x, y)
    img_points[..., 1] = inp_data.mask.shape[0] - img_points[..., 1]

    print("[Step 1] Estimating initial pose.")
    pnp_output = solve_pnp_ransac(
        mesh_features=inp_data.shape_features,
        img_features=inp_data.img_features,
        mesh_verts=inp_data.mesh_verts,
        n_components=opt.pose.n_components,
        img_points=img_points,
        camera_matrix=inp_data.cam_params.K,
        ransac_pnp=opt.pose.ransac_pnp,
    )
    ext_R = pnp_output.ext_Rs[0].float()
    ext_t = pnp_output.ext_ts[0].float()

    inp_data.mesh_matches = inp_data.mesh_verts[pnp_output["verts_idx"].int()]

    ############################################################
    # ! Step 1.2: Run candidate pose fitting
    renderer = NVDiffRenderer(backend=opt.device, device="cuda")

    edge_detector = EdgeDetector().to(opt.device)

    init_values.focal_length_x = focal_x
    init_values.focal_length_y = focal_y

    inp_data = inp_data.to(opt.device)

    orient_loss, iou = [], []
    params = []
    for idx, (ext_R, ext_t) in enumerate(zip(pnp_output["ext_Rs"], pnp_output["ext_ts"])):
        focal_x = inp_data.cam_params.K[0, 0]
        focal_y = inp_data.cam_params.K[1, 1]

        camera = Camera(
            focal_length_x=focal_x,
            focal_length_y=focal_y,
            center=inp_data.cam_params.center,
            rotation=ext_R,
            translation=ext_t,
            dtype=torch.float32,
            cam_type="perspective",
        ).to(opt.device)

        vars = EasierDict(
            rotation=matrix_to_rot6d(torch.eye(3, dtype=torch.float32, device=opt.device))[0],
            translation=torch.zeros(3, dtype=torch.float32, device=opt.device),
            scale=torch.ones(3, dtype=torch.float32, device=opt.device),
        )
        for _, v in vars.items():
            v.requires_grad = True
        optimizer, scheduler = init_optimizer(vars, opt.init_stage.vars)

        with FittingMonitor(
            opt,
            mkdir(save_root / f"orient_{idx + 1:02d}"),
            max_iters=opt.init_stage.max_iters,
            stage_name=f"Orient_{idx + 1:02d}",
            log_debug=False,
        ) as monitor:
            closure = monitor.create_fitting_closure(
                latent=None,
                optimizer=optimizer,
                shape_prior=None,
                mesh_v=inp_data.mesh_verts,
                mesh_f=inp_data.mesh_faces,
                contour_finder=edge_detector,
                rotation=vars.rotation,
                translation=vars.translation,
                scale=vars.scale,
                targets=inp_data,
                renderer=renderer,
                camera=camera,
                resolution=resolution,
                loss_fn=Loss(
                    {
                        "part_mask": 1.0e-4,
                        "part_depth": 1.0e01,
                        "depth": 1.0e0,
                        "mask": 1.0e-3,
                        "mask_iou": 5.0e1,
                        "dist": 1.0e2,
                    }
                ),
            )
            loss = monitor.run_fitting(optimizer, closure, scheduler)
            orient_loss.append(loss[-1].depth + 100 * loss[-1].mask_iou)
            iou.append((1 - loss[-1].mask_iou).item())
            params.append(vars.clone())

    best_orient_idx = torch.argmin(torch.tensor(orient_loss))
    best_orient_loss = orient_loss[best_orient_idx]
    print(f"Best orientation pose {best_orient_idx + 1:d}: {best_orient_loss:02f}")
    print(f"\tOrient losses: {[p.item() for p in orient_loss]}", end=SEPARATOR)

    ############################################################
    # ! Step 2: OPS Fitting
    init_values.update(params[best_orient_idx].detach().cpu())

    ext_R = pnp_output["ext_Rs"][best_orient_idx].float()
    ext_t = pnp_output["ext_ts"][best_orient_idx].float()

    init_values.latent = inp_data.init_latent
    camera = Camera(
        focal_length_x=focal_x,
        focal_length_y=focal_y,
        center=inp_data.cam_params.center,
        rotation=ext_R,
        translation=ext_t,
        dtype=torch.float32,
        cam_type="perspective",
    ).to(opt.device)

    print(f"[Step 2] Starting OPS fitting from candidate pose {best_orient_idx + 1:d}.")
    vars = EasierDict(
        rotation=to_tensor(init_values.rotation),
        translation=to_tensor(init_values.translation),
        scale=to_tensor(init_values.scale),
        latent=to_tensor(init_values.latent),
    ).to(opt.device)
    for _, v in vars.items():
        v.requires_grad = True
    optimizer, scheduler = init_optimizer(vars, opt.shape_stage.vars)

    loss_fn = Loss(opt.shape_stage.loss_cfg)

    shape_prior = Shaper(
        pretrained_exp_dir=opt.shape_prior.exp_dir,
        checkpoint=opt.shape_prior.checkpoint,
        mesh_res=opt.shape_prior.resolution,
        flexi_scale=opt.shape_prior.flexi_scale,
        device=opt.device,
    )

    with FittingMonitor(
        opt,
        mkdir(save_root / "ops_fitting"),
        max_iters=opt.shape_stage.max_iters,
        stage_name="Pose+Shape Fitting",
    ) as monitor:
        closure = monitor.create_fitting_closure(
            optimizer=optimizer,
            shape_prior=shape_prior,
            contour_finder=edge_detector,
            ########################################
            latent=vars.latent,
            rotation=vars.rotation,
            translation=vars.translation,
            scale=vars.scale,
            ########################################
            targets=inp_data,
            renderer=renderer,
            camera=camera,
            loss_fn=loss_fn,
            resolution=resolution,
        )
        loss = monitor.run_fitting(optimizer, closure, scheduler)

    vars = vars.detach().cpu()
    print("Extracting final mesh.")
    img_name = Path(opt.data.input_path).stem
    mesh_name = save_root / f"{img_name}_sdfit"
    with torch.no_grad():
        create_mesh_octree(
            shape_prior.sdf_decoder,
            vars.latent.to(opt.device),
            str(mesh_name),
            N=opt.shape_prior.final_res,
            max_batch=int(2**17),
            clamp_func=lambda x: torch.clamp(x, -0.5, 0.5),
        )

    print(f"Result saved to {save_root.resolve()}", end=SEPARATOR)


if __name__ == "__main__":
    fix_seeds(42)
    cfg = load_config()
    main(cfg)
