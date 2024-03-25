import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .. import utils
from ..data.instance import BopInstanceDataset
from ..data.multi_view_crops import MultiViewCropper
from ..dep.cosy_multiview import get_multiview_frame_index, build_frame_index
from ..data.config import config
from ..data.obj import load_objs
from ..data.renderer import ObjCoordRenderer
from ..surface_embedding import SurfaceEmbeddingModel
from .. import pose_est
from .. import pose_refine
from .. import pose_est_multiview

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--res-data', type=int, default=256)
parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument('--max-poses', type=int, default=10000)
parser.add_argument('--n-pose-evals', type=int, default=1000)
parser.add_argument('--n-mask-samples', type=int, default=5000)
parser.add_argument('--n-2d-3d-corr', type=int, default=5)
parser.add_argument('--n-epi-corr', type=int, default=5)
parser.add_argument('--triplet-sample-factor', type=int, default=1)
parser.add_argument('--n-views', type=int, default=4)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--debug-3d-3d-corr', action='store_true')
parser.add_argument('--filter-obj-id', type=int)
parser.add_argument('--filter-scene-id', type=int)
parser.add_argument('--filter-n-views', type=int)
parser.add_argument('--filter-group-idx', type=int)
parser.add_argument('--filter-pose-idx', type=int)

args = parser.parse_args()
debug = args.debug
debug_3d_3d_corr = args.debug_3d_3d_corr
n_views = args.n_views
n_pose_evals = args.n_pose_evals
n_mask_samples = args.n_mask_samples
n_2d_3d_corr = args.n_2d_3d_corr
n_epi_corr = args.n_epi_corr
triplet_sample_factor = args.triplet_sample_factor
res_crop = args.res_crop
device = torch.device(args.device)
model_path = Path(args.model_path)
assert model_path.is_file()
model_name = model_path.name.split('.')[0]
eval_name = f'{model_name}-gt-multiview-{n_views}'
dataset = model_name.split('-')[0]

results_dir = Path('data/results')
results_dir.mkdir(exist_ok=True)
poses_fp = results_dir / f'{eval_name}-poses.npy'
poses_scores_fp = results_dir / f'{eval_name}-poses-scores.npy'
poses_timings_fp = results_dir / f'{eval_name}-poses-timings.npy'
for fp in poses_fp, poses_scores_fp, poses_timings_fp:
    assert not fp.exists()

# load model
model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_path)).eval().to(device)  # type: SurfaceEmbeddingModel
model.freeze()

# load data
root = Path('data/bop') / dataset
cfg = config[dataset]
test_folder = root / cfg.test_folder
objs, obj_ids = load_objs(root / cfg.model_folder)
assert len(obj_ids) > 0
surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)

data = BopInstanceDataset(root, pbr=False, test=True, cfg=cfg, obj_ids=obj_ids)
insts_by_scene_img = defaultdict(lambda: [])
for inst in data.instances:
    insts_by_scene_img[(inst['scene_id'], inst['img_id'])].append(inst)

multi_view_cropper = MultiViewCropper(objs, data, crop_res=res_crop)
renderer = ObjCoordRenderer(objs, res_crop)
hires_scale = 1
hires_renderer = ObjCoordRenderer(objs, res_crop * hires_scale)

# infer
all_poses = np.empty((2, len(data), 3, 4))
all_scores = np.ones(len(data)) * -np.inf
time_forward, time_pnpransac, time_refine = [], [], []
frame_index = get_multiview_frame_index(build_frame_index(test_folder), n_views=n_views)

inst_count = []


def to_device(t):
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    return t.float().to(device)


csv_lines = ['scene_id,im_id,obj_id,score,R,t,time\n']
csv_lines_refine = csv_lines.copy()

pbar = tqdm(total=len(data))
for group_idx, group in frame_index.iterrows():
    scene_id = group.scene_id
    insts_by_id = defaultdict(lambda: [])
    for view_id in group.view_ids:
        for inst in insts_by_scene_img[(scene_id, view_id)]:
            insts_by_id[inst['pose_idx']].append(inst)
    for pose_idx, insts in insts_by_id.items():
        n_views = len(insts)
        pbar.update(n_views)

        inst_count.append(n_views)
        view_idxs = [group.view_ids.index(inst['img_id']) for inst in insts]
        cams_t_world = group.cams_t_world[view_idxs]
        world_t_cams = np.linalg.inv(cams_t_world)
        Ks = group.Ks[view_idxs]
        view_ids = np.array(group.view_ids)[view_idxs]

        inst = insts[0]
        obj_id = inst['obj_id']

        if debug:
            do_continue = False
            for name, val in ('obj_id', obj_id), ('n_views', n_views), ('group_idx', group_idx), \
                             ('pose_idx', pose_idx), ('scene_id', scene_id):
                arg = getattr(args, f'filter_{name}')
                if arg is not None and arg != val:
                    do_continue = True
            if do_continue:
                continue

        obj_idx = inst['obj_idx']
        obj_ = objs[obj_idx]
        cam_t_obj = np.eye(4)
        cam_t_obj[:3, :3] = inst['cam_R_obj']
        cam_t_obj[:3, 3:] = inst['cam_t_obj']
        world_t_obj_gt = world_t_cams[0] @ cam_t_obj
        # (v, r, r, 3), (v, 3, 3)
        imgs, Ks = multi_view_cropper.get(scene_id=scene_id, world_t_obj=world_t_obj_gt, obj_idx=obj_idx,
                                          img_ids=view_ids, cams_t_world=cams_t_world, Ks=Ks)

        cams_t_world = to_device(cams_t_world)
        world_t_cams = to_device(world_t_cams)
        Ks = to_device(Ks)


        def get_obj_coords(R: torch.tensor, t: torch.tensor, flat=False, renderer=renderer, scale=1.):
            Ks_ = Ks.cpu().numpy().copy()
            Ks_[:2, 2] += 0.5
            Ks_[:2] *= scale
            Ks_[:2, 2] -= 0.5
            cams_R_obj = (cams_t_world[:, :3, :3] @ R).cpu().numpy()
            cams_t_obj = (cams_t_world[:, :3, :3] @ t + cams_t_world[:, :3, 3:]).cpu().numpy()
            return np.stack([
                renderer.render(obj_idx=obj_idx, K=K, R=cam_R_obj, t=cam_t_obj, flat=flat)
                for K, cam_R_obj, cam_t_obj in zip(Ks_, cams_R_obj, cams_t_obj)
            ])


        def get_pose_vis(R: torch.tensor, t: torch.tensor, flat=False):
            vis = imgs / 255  # (v, r, r, 3)
            obj_coords = get_obj_coords(R, t, flat=flat)
            mask = obj_coords[..., 3] == 1.
            alpha = 0.5
            if flat:
                obj_coords[..., (0, 2)] = 0.
            else:
                obj_coords = obj_coords * 0.5 + 0.5
            vis[mask] = (1-alpha) * vis[mask] + alpha * obj_coords[..., :3][mask]
            return utils.image_row(vis)


        with utils.timer('forward', debug):
            # (v, r, r), (v, r, r, e)
            mask_lgts, query_imgs = [torch.stack(t) for t in zip(*[model.infer_cnn(img, obj_idx) for img in imgs])]
            verts = to_device(surface_samples[obj_idx])
            verts_norm = (verts - to_device(obj_.offset)) / obj_.scale
            obj_keys = model.infer_mlp(verts_norm, obj_idx)  # (nk, e)

        if n_views == 1:
            # surfemb pnp (standard)
            with utils.timer('surfemb pnp', debug):
                R, t, scores, *_ = pose_est.estimate_pose(
                    mask_lgts=mask_lgts[0], query_img=query_imgs[0],
                    obj_pts=verts, obj_normals=surface_sample_normals[obj_idx], obj_keys=obj_keys,
                    obj_diameter=obj_.diameter, K=Ks[0], alpha=1.,
                    max_poses=20_000, max_pose_evaluations=5_000,
                )
                R = world_t_cams[0, :3, :3] @ R
                t = world_t_cams[0, :3, :3] @ t + world_t_cams[0, :3, 3:]
        else:
            # surfemb multiview
            with utils.timer('3d3dcorr', debug):
                # (n_samples, n_2d_3d_corr, n_epi_corr, 3), (n_samples, n_2d_3d_corr, 1, 3)
                corr_world, corr_obj = pose_est_multiview.sample_3d_3d_correspondences(
                    mask_lgts=mask_lgts, query_imgs=query_imgs, Ks=Ks, world_t_cams=world_t_cams,
                    obj_pts=verts, obj_keys=obj_keys,
                    n_samples=n_mask_samples, n_2d_3d_corr=n_2d_3d_corr, n_epi_corr=n_epi_corr,
                    batch_size=1_000, debug=debug_3d_3d_corr,
                )

            with utils.timer('triplet+kabsch', debug):
                # (2, m, 3corr, 3xyz), (m,)
                triplets, tri_loss = pose_est_multiview.sample_corr_triplets(corr_world, corr_obj,
                                                                             sample_factor=triplet_sample_factor)
                triplets = triplets[:, :n_pose_evals]
                R, t = pose_est_multiview.kabsch(triplets[1].mT, triplets[0].mT)

            with utils.timer('score', debug):
                scores = torch.stack([
                    pose_est.score_poses(
                        R=cam_t_world[:3, :3] @ R, t=cam_t_world[:3, :3] @ t + cam_t_world[:3, 3:],
                        mask_lgts=mask_lgt, query_img=query_img, K=K, down_sample_scale=3,
                        obj_keys=obj_keys, obj_pts=verts,
                    )[0] for mask_lgt, query_img, K, cam_t_world in zip(mask_lgts, query_imgs, Ks, cams_t_world)
                ]).mean(dim=0)

        # take best scoring pose
        idx_max = scores.argmax()
        R, t, score = R[idx_max], t[idx_max], scores[idx_max]

        with utils.timer('refine', debug):
            R_ref, t_ref, _ = pose_refine.refine_pose(
                world_R_obj=R.cpu().numpy(), world_t_obj=t.cpu().numpy(),
                cams_T_world=cams_t_world.cpu().numpy(), K_crops=Ks.cpu().numpy(),
                query_imgs=query_imgs, mask_lgts=mask_lgts, renderer=renderer, obj_idx=obj_idx,
                obj_=obj_, model=model, keys_verts=obj_keys,
            )
        R_ref, t_ref = to_device(R_ref), to_device(t_ref)

        for R_, t_, lines in (R, t, csv_lines), (R_ref, t_ref, csv_lines_refine):
            cams_R_obj = (cams_t_world[:, :3, :3] @ R_).cpu().numpy()
            cams_t_obj = (cams_t_world[:, :3, :3] @ t_ + cams_t_world[:, :3, 3:]).cpu().numpy()
            for view_id, cam_R_obj, cam_t_obj in zip(view_ids, cams_R_obj, cams_t_obj):
                lines.append(utils.csv_line(
                    scene_id=scene_id, view_id=view_id, obj_id=obj_id,
                    score=score, R=cam_R_obj, t=cam_t_obj, time=0,
                ))

        if debug:
            print('\ngroup,pose,scene,obj', group_idx, pose_idx, scene_id, obj_id)
            print('ref angle [deg]', utils.rotation_magnitude(R.mT @ R_ref) / torch.pi * 180)

            cv2.imshow('inp', utils.image_row(imgs)[..., ::-1])
            cv2.imshow('mask', utils.image_row(torch.sigmoid(mask_lgts)[..., None])[..., 0].cpu().numpy())
            cv2.imshow('queries', utils.image_row(model.get_emb_vis(query_imgs)).cpu().numpy()[..., ::-1])

            #world_t_obj_gt = to_device(world_t_obj_gt)
            #obj_coords_img = get_obj_coords(world_t_obj_gt[:3, :3], world_t_obj_gt[:3, 3:])
            obj_coords_img = get_obj_coords(R_ref, t_ref, renderer=hires_renderer, scale=hires_scale)
            obj_flat_render = get_obj_coords(R_ref, t_ref, flat=True, renderer=hires_renderer, scale=hires_scale)
            obj_coords_mask = obj_coords_img[..., 3] == 1.
            obj_keys_flat = model.infer_mlp(obj_coords_img[obj_coords_mask, :3], obj_idx)

            def query_mouse_cb(ev, x, y, flags, param):
                v_idx = x // res_crop
                x = x % res_crop
                q = query_imgs[v_idx, y, x]  # (e,)
                probs = obj_keys_flat @ (q / 2)
                probs = probs.sub_(probs.max()).exp_().cpu().numpy()[..., None]
                #vis = obj_coords_img[..., :3] * 0.5 + 0.5
                vis = obj_flat_render[..., :3] * 0.7 + 0.2
                vis[obj_coords_mask] = (1 - probs) * vis[obj_coords_mask] + probs * (1, 0, 0)
                vis[~obj_coords_mask] = 1.
                vis = utils.image_row(vis)[..., ::-1]
                cv2.imshow('2d3dcorr', vis)


            cv2.setMouseCallback('queries', query_mouse_cb)

            R_gt, t_gt = to_device(world_t_obj_gt[:3, :3]), to_device(world_t_obj_gt[:3, 3:])
            cv2.imshow('pose', np.concatenate((
                get_pose_vis(R, t, flat=True),
                get_pose_vis(R_ref, t_ref, flat=True),
                get_pose_vis(R_gt, t_gt, flat=True),
            ))[..., ::-1])

            while True:
                key = cv2.waitKey()
                if key == ord('q'):
                    quit()
                elif key == ord(' '):
                    break

print('n_views count', np.unique(inst_count, return_counts=True))
if not debug:
    with open('csv_lines.csv', 'w') as f:
        f.writelines(csv_lines)
    with open('csv_lines_refine.csv', 'w') as f:
        f.writelines(csv_lines_refine)
