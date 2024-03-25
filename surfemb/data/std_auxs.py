from typing import Set

import cv2
import numpy as np

from .instance import BopInstanceDataset, BopInstanceAux
from .tfms import normalize


class RgbLoader(BopInstanceAux):
    def __init__(self, copy=False):
        self.copy = copy

    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        scene_id, img_id = inst['scene_id'], inst['img_id']
        fp = dataset.data_folder / f'{scene_id:06d}/{dataset.img_folder}/{img_id:06d}.{dataset.img_ext}'
        rgb = cv2.imread(str(fp), cv2.IMREAD_COLOR)[..., ::-1]
        assert rgb is not None
        inst['rgb'] = rgb.copy() if self.copy else rgb
        return inst


class MaskLoader(BopInstanceAux):
    def __init__(self, mask_type='mask_visib'):
        self.mask_type = mask_type

    def __call__(self, inst: dict, dataset: BopInstanceDataset) -> dict:
        scene_id, img_id, pose_idx = inst['scene_id'], inst['img_id'], inst['pose_idx']
        mask_folder = dataset.data_folder / f'{scene_id:06d}' / self.mask_type
        mask = cv2.imread(str(mask_folder / f'{img_id:06d}_{pose_idx:06d}.png'), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        inst[self.mask_type] = mask
        return inst


class RandomRotatedMaskCrop(BopInstanceAux):
    def __init__(self, crop_res: int, crop_scale=1.2, max_angle=np.pi,
                 offset_scale=1., use_bbox=False, bbox_from_hull=False,
                 rgb_interpolation=(cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC)):
        self.crop_res, self.crop_scale = crop_res, crop_scale
        self.max_angle = max_angle
        self.rgb_interpolation = rgb_interpolation
        self.offset_scale = offset_scale
        self.use_bbox = use_bbox
        self.bbox_from_hull = bbox_from_hull
        self.definition_aux = RandomRotatedMaskCropDefinition(self)
        self.apply_aux = RandomRotatedMaskCropApply(self)

    def __call__(self, inst: dict, _) -> dict:
        inst = self.definition_aux(inst, _)
        inst = self.apply_aux(inst, _)
        return inst


class RandomRotatedMaskCropDefinition(BopInstanceAux):
    def __init__(self, parent: RandomRotatedMaskCrop):
        self.p = parent

    def __call__(self, inst: dict, _) -> dict:
        theta = np.random.uniform(-self.p.max_angle, self.p.max_angle)
        S, C = np.sin(theta), np.cos(theta)
        R = np.array((
            (C, -S),
            (S, C),
        ))

        if self.p.use_bbox:
            left, top, right, bottom = inst['bbox']
        else:
            if self.p.bbox_from_hull:
                mask_pts = inst['img_convex_hull']  # (N, 2xy)
            else:
                mask_pts = np.argwhere(inst['mask_visib'])[:, ::-1]  # (N, 2xy)

            mask_pts_rotated = mask_pts @ R.T
            left, top = mask_pts_rotated.min(axis=0)
            right, bottom = mask_pts_rotated.max(axis=0)
        inst['used_bbox'] = left, top, right, bottom
        cy, cx = (top + bottom) / 2, (left + right) / 2

        # detector crops can probably be simulated better than this
        size = self.p.crop_res / max(bottom - top, right - left) / self.p.crop_scale
        size = size * np.random.uniform(1 - 0.05 * self.p.offset_scale, 1 + 0.05 * self.p.offset_scale)
        r = self.p.crop_res
        M = np.concatenate((R, [[-cx], [-cy]]), axis=1) * size
        M[:, 2] += r / 2

        offset = (r - r / self.p.crop_scale) / 2 * self.p.offset_scale
        M[:, 2] += np.random.uniform(-offset, offset, 2)
        Ms = np.concatenate((M, [[0, 0, 1]]))

        # calculate axis aligned bounding box in the original image of the rotated crop
        crop_corners = np.array(((0, 0, 1), (0, r, 1), (r, 0, 1), (r, r, 1))) - (0.5, 0.5, 0)  # (4, 3)
        crop_corners = np.linalg.inv(Ms) @ crop_corners.T  # (3, 4)
        crop_corners = crop_corners[:2] / crop_corners[2:]  # (2, 4)
        left, top = np.floor(crop_corners.min(axis=1)).astype(int)
        right, bottom = np.ceil(crop_corners.max(axis=1)).astype(int) + 1
        left, top = np.maximum((left, top), 0)
        right, bottom = np.maximum((right, bottom), (left + 1, top + 1))
        inst['AABB_crop'] = left, top, right, bottom

        inst['M_crop'] = M
        inst['K_crop'] = Ms @ inst['K']
        return inst


class RandomRotatedMaskCropApply(BopInstanceAux):
    def __init__(self, parent: RandomRotatedMaskCrop):
        self.p = parent

    def __call__(self, inst: dict, _) -> dict:
        r = self.p.crop_res
        M = inst['M_crop']
        if 'rgb' in inst:
            inst[f'rgb_crop'] = cv2.warpAffine(inst['rgb'], M, (r, r), flags=np.random.choice(self.p.rgb_interpolation))
        if 'mask_visib' in inst:
            mask_crop = cv2.warpAffine(inst['mask_visib'], M, (r, r), flags=cv2.INTER_LINEAR)
            inst[f'mask_visib_crop'] = (mask_crop / 255).astype(np.float32)
        return inst


class TransformsAux(BopInstanceAux):
    def __init__(self, tfms, key='rgb_crop', crop_key=None):
        self.key = key
        self.tfms = tfms
        self.crop_key = crop_key

    def __call__(self, inst: dict, _) -> dict:
        if self.crop_key is not None:
            left, top, right, bottom = inst[self.crop_key]
            img_slice = slice(top, bottom), slice(left, right)
        else:
            img_slice = slice(None)
        img = inst[self.key]
        img[img_slice] = self.tfms(image=img[img_slice])['image']
        return inst


class NormalizeAux(BopInstanceAux):
    def __init__(self, key='rgb_crop', suffix=''):
        self.key = key
        self.suffix = suffix

    def __call__(self, inst: dict, _) -> dict:
        inst[f'{self.key}{self.suffix}'] = normalize(inst[self.key])
        return inst


class KeyFilterAux(BopInstanceAux):
    def __init__(self, keys=Set[str]):
        self.keys = keys

    def __call__(self, inst: dict, _) -> dict:
        return {k: v for k, v in inst.items() if k in self.keys}
