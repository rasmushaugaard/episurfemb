import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ...data.instance import BopInstanceDataset
from ...data.std_auxs import MaskLoader, RandomRotatedMaskCrop, RgbLoader
from ...data.pose_auxs import ConvexHullProjectionAux
from ...data.config import config
from ...data.obj import load_objs

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--bbox-hull', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

dataset = args.dataset
root = Path('data/bop') / dataset
cfg = config[dataset]
bbox_hull = args.bbox_hull

detection_folder = Path(f'data/detection_results/gt_{"hull" if bbox_hull else "visible"}/{dataset}')
detection_folder.mkdir(exist_ok=True, parents=True)

objs, obj_ids = load_objs(root / cfg.model_folder)
auxs = [
    MaskLoader(),
    ConvexHullProjectionAux(objs),
    RandomRotatedMaskCrop(
        crop_res=224, max_angle=0, offset_scale=0, bbox_from_hull=bbox_hull,
    )
]
if args.debug:
    auxs = [RgbLoader(), *auxs]
data = BopInstanceDataset(root, pbr=False, test=True, cfg=cfg, obj_ids=obj_ids, auxs=auxs)

scene_ids = []
view_ids = []
obj_ids = []
bboxes = []
pose_idxs = []
scores = times = np.zeros(len(data.instances))
masks = np.empty((len(data.instances), 224, 224), dtype=np.uint8)

keys = 'scene_id', 'img_id', 'obj_id', 'used_bbox', 'pose_idx'
lists = scene_ids, view_ids, obj_ids, bboxes, pose_idxs
for i, inst in enumerate(tqdm(data)):
    for k, l in zip(keys, lists):
        l.append(inst[k])
    masks[i] = np.round(inst['mask_visib_crop'] * 255)
    if args.debug:
        cv2.imshow('rgb', inst['rgb_crop'])
        cv2.imshow('mask', masks[i])
        if cv2.waitKey() == ord('q'):
            quit()

bboxes = np.asarray(bboxes).astype(float)

list_names = 'scene_ids', 'view_ids', 'obj_ids', 'bboxes', 'scores', 'times', 'masks', 'pose_idxs'
lists = scene_ids, view_ids, obj_ids, bboxes, scores, times, masks, pose_idxs

for l, name in zip(lists, list_names):
    np.save(str(detection_folder / f'{name}.npy'), np.asarray(l))

csv_lines = ['scene_id,im_id,obj_id,score,R,t,time\n']
for inst in tqdm(data):
    csv_lines.append(','.join([
        str(inst['scene_id']), str(inst['img_id']), str(inst['obj_id']), str(0),
        ' '.join(map(str, inst['cam_R_obj'].reshape(-1))),
        ' '.join(map(str, inst['cam_t_obj'].reshape(-1))),
        str(0),
    ]) + '\n')
with open('/data/bop/results/sanity_tless-test_primesense.csv', 'w') as f:
    f.writelines(csv_lines)
