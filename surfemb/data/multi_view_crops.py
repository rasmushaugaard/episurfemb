from typing import Sequence

import numpy as np
import cv2

from . import pose_auxs, std_auxs, obj


class MultiViewCropper:
    def __init__(self, objs: Sequence[obj.Obj], dataset, crop_res=224):
        self.dataset = dataset
        self.rgb_loader = std_auxs.RgbLoader()
        self.hull_proj = pose_auxs.ConvexHullProjectionAux(objs=objs)
        self.crop_aux = std_auxs.RandomRotatedMaskCrop(
            crop_res=crop_res, max_angle=0., offset_scale=0., rgb_interpolation=cv2.INTER_LINEAR,
            bbox_from_hull=True,
        )

    def get(self, scene_id: int, obj_idx: int, world_t_obj: np.ndarray, img_ids: Sequence[int],
            cams_t_world: np.ndarray, Ks: Sequence[np.ndarray]):
        cams_t_obj = cams_t_world @ world_t_obj[None]
        imgs, K_crops = [], []
        for img_id, cam_t_obj, K in zip(img_ids, cams_t_obj, Ks):
            inst = dict(
                scene_id=scene_id, img_id=img_id, obj_idx=obj_idx, K=K,
                cam_R_obj=cam_t_obj[:3, :3], cam_t_obj=cam_t_obj[:3, 3:],
            )
            inst = self.rgb_loader(inst, self.dataset)
            inst = self.hull_proj(inst, None)
            inst = self.crop_aux(inst, None)

            imgs.append(inst['rgb_crop'])
            K_crops.append(inst['K_crop'])
        return np.stack(imgs), np.stack(K_crops)
