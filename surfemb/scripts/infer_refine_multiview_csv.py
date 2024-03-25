import argparse
import collections
from pathlib import Path

import cv2
import numpy as np
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
parser.add_argument('csv')
parser.add_argument('model')
parser.add_argument('--n-views', type=int, default=4)
parser.add_argument('--n-iter', type=int, default=1)
parser.add_argument('--device', required=True)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--i', type=int)
args = parser.parse_args()

device = torch.device(args.device)
csv_lines = open(args.csv).readlines()[1:]
n_views = args.n_views
model_fp = Path(args.model)
assert model_fp.is_file()
dataset = 'tless'

cfg = config[dataset]
crop_res = 224
root = Path('data/bop') / dataset
test_folder = root / cfg.test_folder
assert root.exists()

model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_fp)).to(device)
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

new_csv_lines = ['scene_id,im_id,obj_id,score,R,t,time\n']
for i in tqdm(range(len(csv_lines))):
    if args.debug:
        i = np.random.randint(len(csv_lines))
        if args.i is not None:
            i = args.i
        print(i)
    line = csv_lines[i].split(',')
    scene_id, main_img_id, obj_id = map(int, line[:3])
    score = float(line[3])
    R = np.array(list(map(float, line[4].split(' ')))).reshape(3, 3)
    t = np.array(list(map(float, line[5].split(' ')))).reshape(3, 1)
    obj_idx = obj_id_to_idx[obj_id]

    # keys of dense surface samples
    obj_ = objs[obj_idx]
    verts_np = surface_samples[obj_idx]
    verts = torch.from_numpy(verts_np).float().to(device)
    # normals = surface_sample_normals[obj_idx]
    verts_norm = (verts_np - obj_.offset) / obj_.scale
    keys_verts = model.infer_mlp(torch.from_numpy(verts_norm).float().to(model.device), obj_idx)  # (N, emb_dim)

    # pose hypothesis before multiview refinement
    main_cam_t_obj = np.eye(4)
    main_cam_t_obj[:3, :3] = R
    main_cam_t_obj[:3, 3:] = t

    # get the set of views from that (scene, view)
    group = multiview_group_index[(scene_id, main_img_id)]
    img_ids = group.view_ids
    main_img_group_idx = img_ids.index(main_img_id)
    cams_t_world = group.cams_t_world

    world_t_cams = np.linalg.inv(cams_t_world)
    world_t_obj = world_t_cams[main_img_group_idx] @ main_cam_t_obj

    imgs, Ks = multi_view_cropper.get(scene_id=scene_id, img_ids=img_ids, obj_idx=obj_idx,
                                      world_t_obj=world_t_obj, cams_t_world=cams_t_world, Ks=group.Ks)

    mask_lgts, query_imgs = zip(*[model.infer_cnn(img, obj_idx) for img in imgs])

    R, t = world_t_obj[:3, :3], world_t_obj[:3, 3:]
    for _ in range(args.n_iter):
        R, t, _ = pose_refine.refine_pose(
            world_R_obj=R, world_t_obj=t,
            cams_T_world=cams_t_world, query_imgs=query_imgs, renderer=renderer, obj_idx=obj_idx,
            obj_=obj_, K_crops=Ks, model=model, keys_verts=keys_verts, mask_lgts=mask_lgts,
        )
    world_t_obj_refined = np.eye(4)
    world_t_obj_refined[:3, :3] = R
    world_t_obj_refined[:3, 3:] = t
    main_cam_t_obj_refined = cams_t_world[main_img_group_idx] @ world_t_obj_refined

    new_csv_lines.append(utils.csv_line(
        scene_id=scene_id, view_id=main_img_id, obj_id=obj_id, score=score,
        R=main_cam_t_obj_refined[:3, :3], t=main_cam_t_obj_refined[:3, 3:], time=0,
    ))

    if args.debug:
        print(score)
        print(main_img_group_idx)
        print()
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

if not args.debug:
    csv_path = Path(args.csv)
    csv_path = csv_path.parent / ('mvref-' + csv_path.name)
    with csv_path.open('w') as f:
        f.writelines(new_csv_lines)
