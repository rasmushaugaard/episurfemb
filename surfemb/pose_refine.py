from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.optimize import minimize

from .data.obj import Obj
from .data.renderer import ObjCoordRenderer
from .utils import Rodrigues
from .surface_embedding import SurfaceEmbeddingModel


def refine_pose(world_R_obj: np.ndarray, world_t_obj: np.ndarray, cams_T_world: np.ndarray,
                query_imgs: Sequence[torch.Tensor],
                renderer: ObjCoordRenderer, obj_idx: int, obj_: Obj,
                K_crops: np.ndarray, model: SurfaceEmbeddingModel,
                keys_verts, mask_lgts=None,
                interpolation='bilinear', n_samples_denom=4096, method='BFGS'):
    """
    Refines the pose estimate (R, t) by local maximization of the log prob (according to the queries / keys)
    of the initially visible surface.
    Bilinear interpolation and PyTorch autograd to get the gradient, and BFGS for optimization.
    """
    n_views = len(query_imgs)
    h, w, _ = query_imgs[0].shape
    assert h == w
    res_crop = h
    device = model.device
    assert n_views == len(cams_T_world) == len(K_crops)

    # Get the object coordinates and keys of the initially visible surface
    world_T_obj = np.eye(4)
    world_T_obj[:3, :3] = world_R_obj
    world_T_obj[:3, 3:] = world_t_obj
    cams_T_obj = cams_T_world @ world_T_obj[None]

    Ks = []
    coords_masked = []
    keys_maskeds = []
    denom_imgs = []
    if mask_lgts is None:
        mask_lgts = [None] * n_views
    for K, cam_T_obj, query_img, mask_lgt in zip(K_crops, cams_T_obj, query_imgs, mask_lgts):
        coord_img = renderer.render(obj_idx, K, cam_T_obj[:3, :3], cam_T_obj[:3, 3:])
        mask = coord_img[..., 3] == 1.
        coord_norm_masked = torch.from_numpy(coord_img[..., :3][mask]).to(device)  # (N, 3)
        keys_masked = model.infer_mlp(coord_norm_masked, obj_idx)  # (N, emb_dim)
        coord_masked = coord_norm_masked * obj_.scale + torch.from_numpy(obj_.offset).to(device)
        coord_masked = torch.cat((coord_masked, torch.ones(len(coord_masked), 1, device=device)), dim=1)  # (N, 4)
        K = torch.from_numpy(K).to(device)

        # precompute log denominator in softmax (log sum exp over keys) per query
        # batched or estimated with reduced amount of keys (as implemented here) because of memory requirements
        keys_sampled = keys_verts[torch.randperm(len(keys_verts), device=device)[:n_samples_denom]]
        denom_img = torch.logsumexp(query_img @ keys_sampled.T, dim=-1, keepdim=True)  # (H, W, 1)
        if mask_lgt is not None:
            denom_img -= F.logsigmoid(mask_lgt)[..., None]

        coord_masked = coord_masked.float()

        Ks.append(K.float())
        coords_masked.append(coord_masked)
        keys_maskeds.append(keys_masked)
        denom_imgs.append(denom_img)

    last_row = torch.tensor((0, 0, 0, 1.)).float().view(1, 4).to(device)
    cams_T_world = torch.from_numpy(cams_T_world).float().to(device)

    def sample(img, p_img_norm):
        # TODO: can't batch sampling normally because there's an uneven number of mask pixels
        #       we could however make it a 3D sampling, which would allow batching.
        samples = F.grid_sample(
            img.permute(2, 0, 1)[None],  # (1, d, H, W)
            p_img_norm[None, None],  # (1, 1, N, 2)
            align_corners=False,
            padding_mode='border',
            mode=interpolation,
        )  # (1, d, 1, N)
        return samples[0, :, 0].T  # (N, d)

    def objective(pose: np.ndarray, return_grad=False):
        pose = torch.from_numpy(pose).float()
        pose.requires_grad = return_grad
        Rt = torch.cat((
            Rodrigues.apply(pose[:3]),
            pose[3:, None],
        ), dim=1).to(device)  # (3, 4)
        Rt = torch.cat((Rt, last_row))  # (4, 4)

        scores = []
        for K, cam_T_world, query_img, coord_masked, keys_masked, denom_img in zip(
                Ks, cams_T_world, query_imgs, coords_masked, keys_maskeds, denom_imgs):
            P = K @ cam_T_world[:3] @ Rt  # (3, 4)
            p_img = coord_masked @ P.T  # (N, 3)
            p_img = p_img[..., :2] / p_img[..., 2:]  # (N, 2)

            # pytorch grid_sample coordinates
            p_img_norm = (p_img + 0.5) * (2 / res_crop) - 1

            query_sampled = sample(query_img, p_img_norm)  # (N, emb_dim)
            log_nominator = (keys_masked * query_sampled).sum(dim=-1)  # (N,)
            log_denominator = sample(denom_img, p_img_norm)[:, 0]  # (N,)
            scores.append(log_nominator - log_denominator)
        score = -torch.cat(scores).mean()

        if return_grad:
            score.backward()
            return pose.grad.detach().cpu().numpy()
        else:
            return score.item()

    rvec = cv2.Rodrigues(world_R_obj)[0]
    pose = np.array((*rvec[:, 0], *world_t_obj[:, 0]))
    result = minimize(fun=objective, x0=pose, jac=lambda pose: objective(pose, return_grad=True), method=method)

    pose = result.x
    R = cv2.Rodrigues(pose[:3])[0]
    t = pose[3:, None]
    return R, t, result.fun
