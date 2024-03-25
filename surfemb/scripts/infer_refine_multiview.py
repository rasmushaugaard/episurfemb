import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np
import pandas
import torch
from tqdm import tqdm

from .. import utils
from ..data.config import config
from ..data.obj import load_objs
from ..data.renderer import ObjCoordRenderer
from ..data.multi_view_crops import MultiViewCropper
from ..surface_embedding import SurfaceEmbeddingModel
from ..dep.cosy_multiview import build_frame_index, get_multiview_frame_index
from .. import pose_refine

parser = argparse.ArgumentParser()
parser.add_argument('poses')
parser.add_argument('--n-views', type=int, default=2)
parser.add_argument('--device', required=True)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

device = torch.device(args.device)
poses_fp = Path(args.poses)
assert poses_fp.name[-10:] == '-poses.npy'
name = poses_fp.name[:-10]
assert name[-5:] == 'depth'  # just for now
n_views = args.n_views

dataset, run, detections = name.split('-')[:3]

all_scene_ids = np.load(f'data/detection_results/{detections}/{dataset}/scene_ids.npy')
all_img_ids = np.load(f'data/detection_results/{detections}/{dataset}/view_ids.npy')
all_obj_ids = np.load(f'data/detection_results/{detections}/{dataset}/obj_ids.npy')
all_pose_idxs = np.load(f'data/detection_results/{detections}/{dataset}/pose_idxs.npy')

cfg = config[dataset]
crop_res = 224
root = Path('data/bop') / dataset
test_folder = root / cfg.test_folder
assert root.exists()

poses = np.load(f'data/results/{name}-poses.npy')
poses_timings = np.load(f'data/results/{name}-poses-timings.npy')
poses_refined = poses.copy()
poses_refined_fp = Path(f'data/results/{name}-multiview_refine_{n_views}-poses.npy')
poses_refined_timings_fp = Path(f'data/results/{name}-multiview_refine_{n_views}-poses-timings.npy')
for fp in poses_refined_fp, poses_refined_timings_fp:
    assert not fp.exists()

model = SurfaceEmbeddingModel.load_from_checkpoint(f'data/models/{dataset}-{run}.ckpt').to(device)
model.eval()
model.freeze()

objs, obj_ids = load_objs(root / cfg.model_folder)
obj_id_to_idx = {id: idx for idx, id in enumerate(obj_ids)}

renderer = ObjCoordRenderer(objs=objs, w=crop_res, h=crop_res)

DatasetProxy = collections.namedtuple('DatasetProxy', ('data_folder', 'img_folder', 'img_ext'))
datasetproxy = DatasetProxy(data_folder=test_folder, img_folder=cfg.img_folder, img_ext=cfg.img_ext)
multi_view_cropper = MultiViewCropper(objs=objs, dataset=datasetproxy)

surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)

multiview_frame_index = get_multiview_frame_index(build_frame_index(test_folder), n_views=n_views)
multiview_group_index = {}
for group_idx, group in multiview_frame_index.iterrows():
    for view_id in group.view_ids:
        multiview_group_index[(group.scene_id, view_id)] = group

# create an index (scene_id, view_id) -> row_idxs, TODO: pandas dataframe groupby
row_idxs = collections.defaultdict(lambda: [])
for i, (scene_id, view_id) in enumerate(zip(all_scene_ids, all_img_ids)):
    row_idxs[(scene_id, view_id)].append(i)

"""
  * For each view group
    * group poses by pose_idx to get "3D detections"
    * take the pose with the best mean score across views based on original crops (to get the best initial pose)
    * refine the pose across the views with original crops
    * propagate pose to all views
"""

for group in multiview_frame_index.iterrows():
    scene_id = group.scene_id
    row_idxs_ = np.concatenate([row_idxs[(scene_id, view_id)] for view_id in group.view_ids])
    pose_idx_to_rows = collections.defaultdict(lambda: [])
    for pose_idx, row_idx in zip(all_pose_idxs[row_idxs_], row_idxs_):
        pose_idx_to_rows[pose_idx].append(row_idx)
    for pose_row_idxs in pose_idx_to_rows.values():  # groups of row idxs that belong to the same pose index
        view_ids = all_img_ids[pose_row_idxs]
        cams_t_obj = poses[pose_row_idxs]
        scores = pose_scores[pose_row_idxs]

        view_idxs = [group.view_ids.index(view_id) for view_id in view_ids]
        cams_t_world = group.cams_t_world[view_idxs]


all_refine_timings = [[], []]
for pi in range(2):
    refine_timings = all_refine_timings[pi]
    for i in tqdm(range(len(all_obj_ids))):
        obj_id = all_obj_ids[i]
        obj_idx = obj_id_to_idx[obj_id]
        scene_id = all_scene_ids[i]
        main_img_id = all_img_ids[i]

        # keys of dense surface samples
        obj_ = objs[obj_idx]
        verts_np = surface_samples[obj_idx]
        verts = torch.from_numpy(verts_np).float().to(device)
        # normals = surface_sample_normals[obj_idx]
        verts_norm = (verts_np - obj_.offset) / obj_.scale
        keys_verts = model.infer_mlp(torch.from_numpy(verts_norm).float().to(model.device), obj_idx)  # (N, emb_dim)

        # pose hypothesis before multiview refinement
        main_cam_t_obj = np.eye(4)
        main_cam_t_obj[:3] = poses[pi, i]
        if np.allclose(main_cam_t_obj[:3, 3], 0):  # if initial pose estimate was unsuccessful
            continue

        # get the set of views
        group = multiview_group_index[(scene_id, main_img_id)]
        scene_img_ids = group.view_ids
        main_idx = scene_img_ids.index(main_img_id)
        cams_t_world = group.cams_t_world
        world_t_cams = np.linalg.inv(cams_t_world)
        Ks = group.Ks
        world_t_obj = world_t_cams[main_idx] @ main_cam_t_obj

        imgs, Ks = multi_view_cropper.get(scene_id=scene_id, img_ids=scene_img_ids, obj_idx=obj_idx,
                                          world_t_obj=world_t_obj, cams_t_world=cams_t_world, Ks=Ks)

        with utils.add_timing_to_list(refine_timings):
            query_imgs = []
            for img in imgs:
                mask_lgts, query_img = model.infer_cnn(img, obj_idx)
                query_imgs.append(query_img)

            R, t, _ = pose_refine.refine_pose(
                world_R_obj=world_t_obj[:3, :3], world_t_obj=world_t_obj[:3, 3:],
                cams_T_world=cams_t_world, query_imgs=query_imgs, renderer=renderer, obj_idx=obj_idx,
                obj_=obj_, K_crops=Ks, model=model, keys_verts=keys_verts
            )
            world_t_obj_refined = np.eye(4)
            world_t_obj_refined[:3, :3] = R
            world_t_obj_refined[:3, 3:] = t
        main_cam_t_obj_refined = cams_t_world[0] @ world_t_obj_refined
        poses_refined[pi, i] = main_cam_t_obj_refined[:3]

        if args.debug:
            query_imgs = model.get_emb_vis(torch.cat(query_imgs, dim=1)).cpu().numpy()
            rows = [query_imgs]
            for pose in world_t_obj, world_t_obj_refined:
                row = []
                for img, K, cam_t_world in zip(imgs, Ks, cams_t_world):
                    cam_t_obj = cam_t_world @ pose
                    render = renderer.render(obj_idx=obj_idx, K=K, R=cam_t_obj[:3, :3], t=cam_t_obj[:3, 3:])
                    mask = render[..., 3] > 0.5
                    img = img / 255.
                    img[mask] = 0.5 * img[mask] + 0.25 + 0.25 * render[mask][:, :3]
                    row.append(img)
                rows.append(np.concatenate(row, axis=1))
            rows = np.concatenate(rows, axis=0)
            cv2.imshow('', rows[..., ::-1])
            while True:
                key = cv2.waitKey()
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    quit()

poses_refined_timings = poses_timings + np.array(all_refine_timings)
if not args.debug:
    np.save(str(poses_refined_fp), poses_refined)
    np.save(str(poses_refined_timings_fp), poses_refined_timings)
