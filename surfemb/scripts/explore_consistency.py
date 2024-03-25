"""
Visualize combined poses from multiple views
"""

import json
import collections
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
import distinctipy

from surfemb.dep.cosy_multiview import build_frame_index, get_multiview_frame_index

base_folder = Path('/data/bop/tless/test_primesense')
frame_index = build_frame_index(base_folder)
frame_index = get_multiview_frame_index(frame_index, n_views=4)

scene_ids = np.load('../data/detection_results/gt_hull_correct/tless/scene_ids.npy')
view_ids = np.load('../data/detection_results/gt_hull_correct/tless/view_ids.npy')
obj_ids = np.load('../data/detection_results/gt_hull_correct/tless/obj_ids.npy')
poses = np.load('../data/results/tless-1ibj6daq-gt_hull_correct-depth-multiview_refine_4-poses.npy')[1]
#poses = np.load('../data/results/tless-1ibj6daq-gt_hull_correct-depth-poses.npy')[0]
#poses = np.load('../data/results/tless-fj')[0]
#poses = np.load('../data/results/tless-1ibj6daq-gt_hull_correct-poses.npy')[0]
pose_scores = np.load('../data/results/tless-1ibj6daq-gt_hull_correct-poses-scores.npy')
poses = np.concatenate((poses, np.zeros((len(poses), 1, 4))), axis=1)
poses[:, -1, -1] = 1

pose_dict = collections.defaultdict(lambda: [])
for scene_id, view_id, obj_id, pose, score in zip(scene_ids, view_ids, obj_ids, poses, pose_scores):
    pose_dict[(scene_id, view_id)].append((obj_id, pose, score))

for i, view_group in frame_index.iterrows():
    scene_id = view_group.scene_id
    if scene_id != 20 or i in {247}:
        continue
    print(i)
    fig = go.Figure()
    view_colors = distinctipy.get_colors(n_colors=view_group.n_views)

    view_id = view_group.view_ids[0]
    gt_poses = []
    for d in json.load(open(f'/data/bop/tless/test_primesense/{scene_id:06d}/scene_gt.json'))[str(view_id)]:
        cam_t_obj = np.eye(4)
        cam_t_obj[:3, :3] = np.array(d['cam_R_m2c']).reshape(3, 3)
        cam_t_obj[:3, 3] = d['cam_t_m2c']
        gt_poses.append(cam_t_obj)
    gt_poses = np.stack(gt_poses)
    gt_poses = np.linalg.inv(view_group.cams_t_world[0]) @ gt_poses

    world_t_objs = []
    world_t_cams = []
    scores = []
    for view_id, cam_t_world, view_color in zip(view_group.view_ids, view_group.cams_t_world, view_colors):
        world_t_cam = np.linalg.inv(cam_t_world)
        world_t_cams.append(world_t_cam)
        for obj_id, cam_t_obj, score in pose_dict[(scene_id, view_id)]:
            world_t_obj = world_t_cam @ cam_t_obj
            scores.append(score)
            world_t_objs.append(world_t_obj)
            line_c2o = np.stack((world_t_cam[:3, 3], world_t_obj[:3, 3])).T  # (3, 2)
            fig.add_scatter3d(mode='lines', x=line_c2o[0], y=line_c2o[1], z=line_c2o[2],
                              line=dict(color='black'), hoverinfo='skip')
    world_t_objs = np.stack(world_t_objs)
    world_t_cams = np.stack(world_t_cams)

    fig.add_scatter3d(
        mode='markers',
        x=world_t_objs[:, 0, 3],
        y=world_t_objs[:, 1, 3],
        z=world_t_objs[:, 2, 3],
        hovertext=scores,
        marker=dict(color='rgba(0, 0, 0, 0)', line=dict(width=2, color='blue'))
    )
    fig.add_scatter3d(
        mode='markers',
        x=gt_poses[:, 0, 3],
        y=gt_poses[:, 1, 3],
        z=gt_poses[:, 2, 3],
        marker=dict(color='green')
    )
    fig.add_scatter3d(
        mode='markers',
        x=world_t_cams[:, 0, 3],
        y=world_t_cams[:, 1, 3],
        z=world_t_cams[:, 2, 3],
        hovertext=view_group.view_ids,
        marker=dict(color='black'),
    )

    fig.update_layout(scene=dict(aspectmode='data'), xaxis=dict(ticks=''))
    fig.show()
    quit()
