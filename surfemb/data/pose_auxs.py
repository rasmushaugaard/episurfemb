from typing import Sequence

import numpy as np

from .instance import BopInstanceAux
from .obj import Obj
from .renderer import ObjCoordRenderer


class ObjCoordAux(BopInstanceAux):
    def __init__(self, objs: Sequence[Obj], res: int, sigma=0.):
        self.objs, self.res = objs, res
        self.renderer = None
        self.sigma = sigma

    def get_renderer(self):
        # lazy instantiation of renderer to create the context in the worker process
        if self.renderer is None:
            self.renderer = ObjCoordRenderer(self.objs, self.res)
        return self.renderer

    def __call__(self, inst: dict, _) -> dict:
        renderer = self.get_renderer()
        K = inst['K_crop'].copy()

        if self.sigma > 0:
            # offset principal axis slightly to encourage all object coordinates within the pixel to have
            # som probability mass. Smoother probs -> more robust score and better posed refinement opt. problem.
            while True:
                offset = np.random.randn(2)
                if np.linalg.norm(offset) < 3:
                    K[:2, 2] += offset * self.sigma
                    break

        obj_coord = renderer.render(inst['obj_idx'], K, inst['cam_R_obj'], inst['cam_t_obj']).copy()
        inst['obj_coord_mask'] = obj_coord[..., 3] == 1.
        inst['obj_coord'] = obj_coord
        return inst


class SurfaceSampleAux(BopInstanceAux):
    def __init__(self, objs: Sequence[Obj], n_samples: int, norm=True):
        self.objs, self.n_samples = objs, n_samples
        self.norm = norm

    def __call__(self, inst: dict, _) -> dict:
        obj = self.objs[inst['obj_idx']]
        mesh = obj.mesh_norm if self.norm else obj.mesh
        inst['surface_samples'] = mesh.sample(self.n_samples).astype(np.float32)
        return inst


class MaskSamplesAux(BopInstanceAux):
    def __init__(self, n_samples: int, mask_key='obj_coord_mask'):
        self.mask_key = mask_key
        self.n_samples = n_samples

    def __call__(self, inst: dict, _):
        mask_arg = np.argwhere(inst[self.mask_key])  # (N, 2)
        idxs = np.random.choice(np.arange(len(mask_arg)), self.n_samples, replace=self.n_samples > len(mask_arg))
        inst['mask_samples'] = mask_arg[idxs]  # (n_samples, 2)
        return inst


class ConvexHullProjectionAux(BopInstanceAux):
    def __init__(self, objs: Sequence[Obj]):
        self.objs = objs

    def __call__(self, inst: dict, _) -> dict:
        obj = self.objs[inst['obj_idx']]
        cam_pts = inst['cam_R_obj'] @ obj.mesh.convex_hull.vertices.T + inst['cam_t_obj']  # (3, N)
        img_pts = inst['K'] @ cam_pts
        img_pts = img_pts[:2] / img_pts[2:]
        inst['img_convex_hull'] = img_pts.T  # (N, 2)
        return inst


class RandomMaskFillAux(BopInstanceAux):
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, inst: dict, _) -> dict:
        if np.random.rand() < self.p:
            rgb = inst['rgb'].astype(np.uint16)
            mask = inst['mask_visib']
            arg_mask = np.argwhere(mask > 128)  # (n, 2yx)
            y, x = arg_mask[np.random.randint(len(arg_mask))]
            rgb = mask[..., None] * rgb[y, x] + (255 - mask[..., None]) * rgb
            inst['rgb'] = (rgb // 255).astype(np.uint8)
        return inst
