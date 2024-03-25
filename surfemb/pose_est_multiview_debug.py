import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.graph_objs as go

from .data.renderer import ObjCoordRenderer
from .surface_embedding import SurfaceEmbeddingModel
from .dep.cosy_multiview import build_frame_index, get_multiview_frame_index
from .data.multi_view_crops import MultiViewCropper
from .data.config import config
from .data.obj import load_objs
from .data.instance import BopInstanceDataset
from . import epipolar
from . import utils
from . import pose_est_multiview
from . import pose_est

scene_id, obj_id, seed = [
    (10, 22, 0),  # no symmetry
    (1, 25, 0),  # 2 degree sym
    (1, 30, 0),  # 4 degree sym
    (20, 1, 0),  # almost cont. sym.
    (16, 14, 1),  # cont. sym
][2]
torch.manual_seed(seed)

n_views = 4
res = 224
dataset = 'tless'
device = ['cpu', 'cuda:0'][1]
cfg = config[dataset]
root = Path('data/bop') / dataset
test_folder = root / cfg.test_folder
frame_index = get_multiview_frame_index(build_frame_index(test_folder), n_views=4)
obj_id_to_idx = {int(k): idx for idx, k in
                 enumerate(json.load((root / 'models_cad' / 'models_info.json').open()).keys())}

objs, obj_ids = load_objs(root / cfg.model_folder, obj_ids=[obj_id])
surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)

model = SurfaceEmbeddingModel.load_from_checkpoint(
    '/hdd/surfemb/models/tless-1ibj6daq.ckpt')  # type: SurfaceEmbeddingModel
model.eval().to(device)
model.freeze()

data = BopInstanceDataset(root, pbr=False, test=True, cfg=cfg, obj_ids=obj_ids, scene_ids=[scene_id])
# DatasetProxy = namedtuple('DatasetProxy', ('data_folder', 'img_folder', 'img_ext'))
# dataset_proxy = DatasetProxy(data_folder=test_folder, img_folder=cfg.img_folder, img_ext=cfg.img_ext)
multi_view_cropper = MultiViewCropper(objs, data, crop_res=res)
renderer = ObjCoordRenderer(objs, res)

# i = np.random.randint(len(data))
i = 0
inst = data[i]
scene_id = inst['scene_id']
view_id = inst['img_id']
obj_idx = inst['obj_idx']
obj_idx_ = obj_id_to_idx[inst['obj_id']]
for group_idx, group in frame_index.iterrows():
    if group_idx != 4:
        continue
    if group.scene_id != scene_id or view_id not in group.view_ids:
        continue
    n_views = group.n_views
    cam_t_obj = np.eye(4)
    cam_t_obj[:3, :3] = inst['cam_R_obj']
    cam_t_obj[:3, 3:] = inst['cam_t_obj']
    cams_t_world = group.cams_t_world
    world_t_cams = np.linalg.inv(cams_t_world)
    main_view_idx = group.view_ids.index(view_id)
    world_t_obj = world_t_cams[main_view_idx] @ cam_t_obj
    cams_t_obj = cams_t_world @ world_t_obj
    imgs, Ks = multi_view_cropper.get(scene_id=scene_id, world_t_obj=world_t_obj, obj_idx=obj_idx,
                                      img_ids=group.view_ids, cams_t_world=group.cams_t_world, Ks=group.Ks)
    mask_lgts, queries = [torch.stack(t) for t in zip(*[model.infer_cnn(img, obj_idx_) for img in imgs])]
    # (v, r, r), (v, r, r, e)
    Rs, ts = cams_t_obj[:, :3, :3], cams_t_obj[:, :3, 3:]
    coords = np.stack([renderer.render(obj_idx, K, R, t) for K, R, t in zip(Ks, Rs, ts)])
    coords = torch.from_numpy(coords).to(device)

    obj_ = objs[obj_idx]
    verts = surface_samples[obj_idx]
    verts_norm = (verts - obj_.offset) / obj_.scale
    obj_keys = model.infer_mlp(torch.from_numpy(verts_norm).float().to(device), obj_idx_)  # (nk, e)

    for _ in range(2):
        with utils.timer('corr'):
            # (n_samples, n_2d_3d_corr, n_epi_corr, 3), (n_samples, n_2d_3d_corr, 1, 3)
            corr_world, corr_obj = pose_est_multiview.sample_3d_3d_correspondences(
                mask_lgts=mask_lgts, query_imgs=queries,
                Ks=torch.from_numpy(Ks).float().to(device),
                world_t_cams=torch.from_numpy(world_t_cams).float().to(device),
                obj_pts=torch.from_numpy(verts).float().to(device),
                obj_keys=obj_keys, n_samples=5_000, n_2d_3d_corr=5, n_epi_corr=5,
                batch_size=1_000, epi_uniform_view_priors=False,
                debug=False, temp_2d_3d=1.,
            )
            corr_obj.view(-1)[-1].item()
    # quit()
    # cv2.waitKey(100)

    # (2, n, 3corr, 3xyz), n
    with utils.timer('sample triplets'):
        corr, loss = pose_est_multiview.sample_corr_triplets(corr_world, corr_obj, sample_factor=1)
        corr.view(-1)[-1].item()
    corr_world_, corr_obj_ = corr
    # world_pts = world_R_obj @ obj_pts + world_t_obj
    # Q = R @ P + t
    with utils.timer('kabsch'):
        R, t = pose_est_multiview.kabsch(P=corr_obj_.mT, Q=corr_world_.mT)
        R.view(-1)[-1].item()
    t_err = torch.norm(t[:, :, 0].cpu() - world_t_obj[:3, 3], dim=1)

    n_pose_eval = 1000
    #R, t = world_t_obj[None, :3, :3], world_t_obj[None, :3, 3:]
    #R, t = [torch.from_numpy(v).float().to(device) for v in (R, t)]
    with utils.timer('pose score'):
        pose_scores = torch.stack([
            pose_est.score_poses(
                R=cam_t_world[:3, :3] @ R[:n_pose_eval],
                t=cam_t_world[:3, :3] @ t[:n_pose_eval] + cam_t_world[:3, 3:],
                mask_lgts=mask_lgt, query_img=query_img, K=K,
                obj_keys=obj_keys, obj_pts=torch.from_numpy(verts).float().to(device), down_sample_scale=3,
                debug=True,
            )[0] for mask_lgt, query_img, K, cam_t_world in
            zip(mask_lgts, queries, torch.from_numpy(Ks).float().to(device),
                torch.from_numpy(cams_t_world).float().to(device))
        ]).mean(dim=0)
        pose_scores[-1].item()
    pose_scores = pose_scores.cpu()
    cv2.waitKey()

    if False:
        fig = go.Figure()
        fig.add_scatter(
            mode='markers',
            marker=dict(color='rgba(255,0,0,0.1)'),
            x=loss.cpu(),
            y=t_err,
        )
        fig.update_xaxes(type='log')
        fig.show()
    if False:
        fig = go.Figure()
        fig.add_scatter(
            mode='markers',
            marker=dict(color='rgba(255,0,0,1)'),
            x=pose_scores,
            y=t_err[:n_pose_eval],
        )
        fig.show()

    world_corr_obj = corr_obj.view(-1, 3) @ world_t_obj[:3, :3].T + world_t_obj[:3, 3]
    world_pts = corr_world.view(-1, 3)

    if False:
        fig = go.Figure()
        fig.add_scatter3d(
            name='cams',
            mode='markers',
            x=world_t_cams[:, 0, 3],
            y=world_t_cams[:, 1, 3],
            z=world_t_cams[:, 2, 3],
            marker=dict(size=20),
        )
        fig.add_scatter3d(
            name='world_pts',
            mode='markers',
            x=world_pts[:, 0],
            y=world_pts[:, 1],
            z=world_pts[:, 2],
            marker=dict(size=2, color='rgba(255,0,0,0.1)'),
        )
        fig.add_scatter3d(
            name='obj_pts',
            mode='markers',
            x=world_corr_obj[:, 0],
            y=world_corr_obj[:, 1],
            z=world_corr_obj[:, 2],
            marker=dict(size=2),
        )
        for line in corr[0, 0], corr[1, 0] @ world_t_obj[:3, :3].T + world_t_obj[:3, 3]:
            fig.add_scatter3d(
                mode='lines',
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                line=dict(color='black'),
            )

        fig.update_layout(
            scene=dict(aspectmode='data'),
            # scene_camera=dict(center={k: v for k, v in zip('xyz', world_t_obj[:3, 3])}),
        )
        fig.show()
        # quit()
        quit()
    # (2, n_samples, n_2d_3d_corr, n_epi_corr, 3xyz, 3corr)
    # pts = torch.stack((corr, corr[torch.randperm(len(corr))], corr[torch.randperm(len(corr))]), dim=-1)
    # pts = pts.view(2, -1, 3, 3)
    # mu = pts.mean(dim=-1, keepdim=True)

    obj_keys = obj_keys[torch.randperm(len(obj_keys))[:2048]]
    denom = torch.logsumexp(queries @ obj_keys.T, dim=-1)  # (v, r, r)
    assert denom.shape == (n_views, res, res)

    imgs = torch.from_numpy(np.stack(imgs)).float() / 255  # (v, r, r, 3)
    # imgs = model.get_emb_vis(queries)
    # imgs = coords[..., :3] * 0.5 + 0.5

    # get all fundamental matrices
    Ps = Ks @ cams_t_world[:, :3]  # (n_views, 3, 4)
    world_p_cams = world_t_cams[:, :, 3:]  # (n_views, 4, 1)
    F = epipolar.get_fundamental_matrices(torch.from_numpy(Ps).float(), torch.from_numpy(world_p_cams).float())
    # (v, r, r, v), 2 x (v, v, 3, nl)
    epi_line_index, epi_lines_s, epi_lines_t = epipolar.build_epipolar_line_index(F, res)
    nl = epi_lines_s.shape[-1]
    xx, yy, mask = epipolar.lines_raster(epi_lines_t, res=res, dim=-2)  # 3 x (v, v, r, nl)
    xx[~mask] = yy[~mask] = 0
    img_lines = imgs[
        torch.arange(n_views).view(1, n_views, 1, 1),
        yy, xx,
    ]  # (v, v, r, nl, 3)
    assert img_lines.shape == (n_views, n_views, res, nl, 3)
    img_lines[~mask] = 0
    query_lines = torch.cat((
        queries, denom[..., None], torch.sigmoid(mask_lgts)[..., None]
    ), dim=-1)[
        torch.arange(n_views).view(1, n_views, 1, 1),
        yy, xx,
    ]  # (v, v, r, nl, e + 2)
    query_lines[~mask] = 0

    imgs = torch.cat(tuple(imgs), dim=1)


    def mouse_cb(ev, x, y, flags, param):
        if y >= res:
            return
        view_idx = x // res
        x = x % res

        epi_idx = epi_line_index[view_idx, y, x]  # (n_views,)
        lines_tgt = epi_lines_t[view_idx, torch.arange(n_views), :, epi_idx]  # (n_views, 3)
        lines_src = epi_lines_s[view_idx, torch.arange(n_views), :, epi_idx]  # (n_views, 3)
        img_lines_ = img_lines[view_idx, torch.arange(n_views), :, epi_idx]  # (n_views, crop_res, 3)
        img_lines_ = img_lines_.view(1, -1, 3).expand(20, -1, -1)

        query_lines_ = query_lines[view_idx, torch.arange(n_views), :, epi_idx]  # (n_views, crop_res, e + 2)
        query_lines_ = query_lines_.view(n_views * res, model.emb_dim + 2)
        query_lines_, denominator, mask_lines = query_lines_[..., :-2], query_lines_[..., -2], query_lines_[..., -1]

        imgs_ = imgs.clone()
        img_views = imgs_.view(res, n_views, res, 3)
        colors = torch.zeros(4, 3)
        colors[(0, 1, 2), (0, 1, 2)] = 1.
        for i in range(n_views):
            c = colors[i]
            epipolar.draw_line(lines_tgt[i], img_views[:, i], c)
            epipolar.draw_line(lines_src[i], img_views[:, view_idx], c)

        k = model.infer_mlp(coords[view_idx, y, x, :3][None], obj_idx_)[0]  # (e,)
        nominator = (query_lines_ * k).sum(dim=-1)  # (v * r)
        probs = (nominator - denominator).exp()
        mask = (query_lines_ == 0).all(dim=-1)
        probs[mask] = 0
        probs = probs * mask_lines
        probs_per_view = probs.view(n_views, res).sum(dim=1).cpu()
        probs = (probs / probs.max()) ** 0.5
        probs = probs.view(1, -1, 1).expand(20, -1, 3)

        imgs_ = torch.cat((imgs_, img_lines_, probs.cpu()), dim=0)
        imgs_ = imgs_.cpu().numpy()[..., ::-1]
        cv2.imshow('imgs', imgs_)


    cv2.imshow('imgs', imgs.cpu().numpy()[..., ::-1])
    cv2.setMouseCallback('imgs', mouse_cb)
    mouse_cb(None, 50, 50, None, None)

    key = cv2.waitKey()
    if key == ord('q'):
        quit()
