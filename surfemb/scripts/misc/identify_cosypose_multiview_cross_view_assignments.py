import queue
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import scipy.optimize
import matplotlib.pyplot as plt

from surfemb.dep.cosy_multiview import get_multiview_frame_index, build_frame_index


def connected_components(edges):
    components = []
    visited = set()
    q = queue.Queue()
    for vert in edges.keys() if isinstance(edges, dict) else range(len(edges)):
        if vert in visited:
            continue
        q.put(vert)
        component = []
        while not q.empty():
            v = q.get()
            if v in visited:
                continue
            visited.add(v)
            component.append(v)
            for vn in edges[v]:
                q.put(vn)
        components.append(component)
    return components


csv_lines = open('/data/bop/results/cosy-tless419066-4views_tless-test_primesense.csv').readlines()[1:]
test_folder = Path('/data/bop/tless/test_primesense')
multiview_frame_index = get_multiview_frame_index(build_frame_index(test_folder), n_views=4)
multiview_group_index = {}
for group_idx, group in multiview_frame_index.iterrows():
    for view_id in group.view_ids:
        multiview_group_index[(group.scene_id, view_id)] = group

poses = defaultdict(lambda: [])
obj_ids = defaultdict(lambda: [])
scores = defaultdict(lambda: [])

for line in csv_lines:
    line = line.split(',')
    scene_id, img_id, obj_id = map(int, line[:3])
    score = float(line[3])
    R = np.array(list(map(float, line[4].split(' ')))).reshape(3, 3)
    t = np.array(list(map(float, line[5].split(' ')))).reshape(3, 1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    key = scene_id, img_id
    poses[key].append(T)
    obj_ids[key].append(obj_id)
    scores[key].append(score)

all_objs_t_objs = {}
for key in poses.keys():
    obj_ids[key] = np.array(obj_ids[key])
    scores[key] = np.array(scores[key])
    cam_t_objs = poses[key] = np.stack(poses[key])
    n_pred = len(cam_t_objs)
    objs_t_cam = np.linalg.inv(cam_t_objs)
    objs_t_objs = objs_t_cam[:, None] @ cam_t_objs[None, :]  # (n_pred, n_pred, 4, 4)
    diag_mask = ~np.eye(n_pred, dtype=bool)
    objs_t_objs = objs_t_objs[diag_mask][:, :3]  # (n_pred * (n_pred - 1), 3, 4)
    pose_idxs = np.argwhere(diag_mask)  # (n_pred * (n_pred - 1), 2)
    all_objs_t_objs[key] = KDTree(objs_t_objs.reshape(-1, 12)), pose_idxs

# Check if any relative pose matches between views.
# That will provide the extrinsics found by cosypose.
new_csv_lines = ['scene_id,im_id,obj_id,score,R,t,time\n']
did_write_line = set()
for group_idx, group in tqdm(multiview_frame_index.iterrows(), total=len(multiview_frame_index)):
    scene_id = group.scene_id
    n_views = group.n_views
    """
    Find extrinsics. If there is ANY exact match of relative poses, it's presumably because both instances have been 
    part of a multiview optimization, and then they inform us about the relative camera poses.
    """
    pred_edges = defaultdict(lambda: set())
    extrinsics = defaultdict(lambda: [])

    view_edges = [[] for _ in range(n_views)]
    for a, view_id_a in enumerate(group.view_ids):
        key_a = scene_id, view_id_a
        cam_a_t_objs, obj_ids_a = poses[key_a], obj_ids[key_a]
        tree_a, pose_indices_a = all_objs_t_objs[key_a]
        for b, view_id_b in enumerate(group.view_ids[a + 1:], start=a + 1):
            key_b = scene_id, view_id_b
            cam_b_t_objs, obj_ids_b = poses[key_b], obj_ids[key_b]
            tree_b, pose_indices_b = all_objs_t_objs[(scene_id, view_id_b)]
            dists, pose_match_a_idx = tree_a.query(tree_b.data)
            pose_indices_a_ = pose_indices_a[pose_match_a_idx]
            mask = dists < 1e-4
            # obj_0_t_cam_a @ cam_a_t_obj_1 == obj_0_t_cam_b @ cam_b_t_obj_1
            for (a0, a1), (b0, b1) in zip(pose_indices_a_[mask], pose_indices_b[mask]):
                # print(a, b, a0, a1, b0, b1)
                # quit()
                # link the predictions across views
                assert obj_ids_a[a0] == obj_ids_b[b0]
                assert obj_ids_a[a1] == obj_ids_b[b1]
                pred_edges[(a, a0)].add((b, b0))
                pred_edges[(b, b0)].add((a, a0))
                pred_edges[(a, a1)].add((b, b1))
                pred_edges[(b, b1)].add((a, a1))
                # and save the associated extrinsic
                extrinsics[(a, b)].append(cam_a_t_objs[a0] @ np.linalg.inv(cam_b_t_objs[b0]))
                view_edges[a].append(b)
                view_edges[b].append(a)

    # sanity check: verify extrinsic consistency
    for (a, b), a_ts_b in extrinsics.items():
        a_ts_b = np.stack(a_ts_b)[:, :3].reshape(-1, 12)
        ma_dist = np.abs(a_ts_b - a_ts_b.mean(axis=0)).max()
        assert ma_dist < 1e-3, ma_dist

    # view_connected_components = connected_components(view_edges)
    pred_connected_components = connected_components(pred_edges)

    group_world_t_cams = np.linalg.inv(group.cams_t_world)
    for conn_comp in pred_connected_components:
        # for each view, project a set of points into the view based on the view-centric pose estimate
        # choose one of the views to get initial world_t_obj and perform bundle adjustment to refine world_t_obj

        # there seem to be a few duplicate poses, which we get rid of here:
        conn_comp = np.array(conn_comp)
        conn_comp = conn_comp[np.unique(conn_comp[:, 0], return_index=True)[1]]
        view_idxs, inst_idxs = conn_comp.T
        keys = [(scene_id, group.view_ids[view_idx]) for view_idx in view_idxs]
        obj_ids_ = [obj_ids[key][inst_idx] for key, inst_idx in zip(keys, inst_idxs)]
        scores_ = [scores[key][inst_idx] for key, inst_idx in zip(keys, inst_idxs)]
        cams_t_inst = np.stack([poses[key][inst_idx] for key, inst_idx in zip(keys, inst_idxs)])

        assert len(np.unique(obj_ids_)) == 1
        obj_id = obj_ids_[0]
        score = np.max(scores_)

        # find world_t_inst with gt extrinsics based on least squares error wrt. view-centric poses with est. extrinsics
        cams_t_world = group.cams_t_world[view_idxs]
        world_t_cams = group_world_t_cams[view_idxs]
        Ks = group.Ks[view_idxs]
        Ps = Ks @ cams_t_world[:, :3]

        obj_pts = np.random.uniform(-100., 100., (50, 3))  # TODO: load vertices
        obj_pts_h = np.concatenate((obj_pts, np.ones((len(obj_pts), 1))), axis=1)


        def project(Ps, world_t_obj):
            pts = (Ps @ world_t_obj) @ obj_pts_h.T  # (n_views, 3, n_pts)
            return pts[:, :2] / pts[:, 2:]  # (n_views, 2, n_pts)


        world_ts_inst = world_t_cams @ cams_t_inst  # (n_views, 4, 4)
        view_centric_pts = project(Ps, world_ts_inst)

        pose_init = np.concatenate((
            Rotation.from_matrix(world_ts_inst[0, :3, :3]).as_rotvec(),
            world_ts_inst[0, :3, 3],
        ))


        def pose_to_transform(pose):
            world_t_inst = np.eye(4)
            world_t_inst[:3, :3] = Rotation.from_rotvec(pose[:3]).as_matrix()
            world_t_inst[:3, 3] = pose[3:]
            return world_t_inst


        res = scipy.optimize.least_squares(
            fun=lambda pose: (view_centric_pts - project(Ps, pose_to_transform(pose))).reshape(-1),
            x0=pose_init
        )
        assert res.success
        world_t_inst = pose_to_transform(res.x)
        cams_t_inst = cams_t_world @ world_t_inst

        # scene_id,im_id,obj_id,score,R,t,time
        for view_idx, inst_idx, cam_t_inst in zip(view_idxs, inst_idxs, cams_t_inst):
            view_id = group.view_ids[view_idx]
            new_csv_lines.append(','.join([
                str(scene_id), str(view_id), str(obj_id), str(score),
                ' '.join(map(str, cam_t_inst[:3, :3].reshape(-1))),
                ' '.join(map(str, cam_t_inst[:3, 3])),
                str(0),
            ]) + '\n')
            did_write_line.add((scene_id, view_id, inst_idx))

print(len(csv_lines), len(new_csv_lines))
quit()
with open('/data/bop/results/wip_tless-test_primesense.csv', 'w') as f:
    f.writelines(new_csv_lines)
