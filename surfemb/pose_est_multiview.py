import cv2
import torch
import torch.nn.functional
from . import epipolar


def closest_point_to_lines(p0: torch.tensor, v0: torch.tensor, p1: torch.tensor, v1: torch.tensor):
    """
    This is approx 20 x times faster than using least squares on 3D-line equations.
    Could probably be faster with epipolar plane cashing and calculating intersection as 2D cross product, before
    elevating them to 3D again, but this implementation is a bit simpler and fast enough to not be the limiting factor.
    """
    v2 = torch.cross(v0, v1)
    # p0 + t0v0 = p1 + t1v1 + t2v2
    # t0v0 - t1v1 - t2v2 = p1 - p0
    A = torch.stack(torch.broadcast_tensors(v0, -v1, -v2), dim=-1)  # (..., 3eq(xyz), 3unknowns)
    b = p1 - p0  # (s, 3eq(xyz))
    t0, t1, t2 = torch.linalg.solve(A, b).split(1, -1)  # (s, 3unknowns)
    p = p1 + t1 * v1 + 0.5 * t2 * v2
    return p


def sample_3d_3d_correspondences(
        mask_lgts: torch.tensor, query_imgs: torch.tensor,
        Ks: torch.tensor, world_t_cams: torch.tensor,
        obj_pts: torch.tensor, obj_keys: torch.tensor,
        n_samples: int, batch_size=5_000, n_2d_3d_corr=1, n_epi_corr=1,
        key_sample_size=2_048, epi_uniform_view_priors=False, debug=False,
        temp_2d_3d=1., epi_view_prior_by_max=False
):
    r"""
    .. math::
        p(q_i|k) :\propto p(k|q_i) p(mask_i) = \exp(k q_i - \sum_j[k_j q_i] + logsigmoid(mlgt_i))

    where only the term :math:`k q_i` depends on k, and the two other terms can be cached.

    We sample 2D points proportional to their mask probabilities across views.
    This favors non-occluded views which could be beneficial, but also favors views with epipolar lines along
    ambiguities, which is not ideal. Try uniform view priors.

    :returns: 3D-3D corr with shape (n_samples, n_2d_3d_corr, n_epi_corr, 2(world, obj), 3(xyz))
    """
    v, r, _, e = query_imgs.shape
    assert query_imgs.shape == (v, r, r, e)
    assert mask_lgts.shape == (v, r, r)
    assert Ks.shape == (v, 3, 3)
    assert world_t_cams.shape == (v, 4, 4)
    m = len(obj_pts)
    assert obj_pts.shape == (m, 3)
    assert obj_keys.shape == (m, e)
    assert not (epi_uniform_view_priors and epi_view_prior_by_max)
    device = mask_lgts.device

    # cache log_base
    obj_keys_denom_sample = obj_keys[torch.randperm(m, device=device)[:key_sample_size]]
    # (v, r, r, 1)
    log_base = (
            torch.nn.functional.logsigmoid(mask_lgts) -
            torch.logsumexp(query_imgs @ obj_keys_denom_sample.T, dim=-1)
    )[..., None]

    # cache rays
    img_pts = torch.stack((
        *torch.meshgrid(torch.arange(r, device=device), torch.arange(r, device=device), indexing='xy'),
        torch.ones((r, r), device=device),
    ), dim=-1)  # (r, r, 3xyw)
    img_rays = (torch.linalg.inv(Ks)[:, None, None] @ img_pts[..., None])[..., 0]  # (v, r, r, 3)
    img_rays = img_rays @ world_t_cams[:, None, :3, :3].mT  # in world coordinates
    assert img_rays.shape == (v, r, r, 3), img_rays.shape

    # cache epipolar lines
    cams_t_world = torch.linalg.inv(world_t_cams)
    Ps = Ks @ cams_t_world[:, :3]  # (v, 3, 4)
    world_p_cams = world_t_cams[:, :, 3:]  # (v, 4, 1)
    F = epipolar.get_fundamental_matrices(Ps, world_p_cams)  # (v, v, 3, 3)
    # (v, r, r, v), 2 x (v, v, 3, nl)
    epi_lines_idx, epi_lines_src, epi_lines_tgt = epipolar.build_epipolar_line_index(F, r)
    nl = epi_lines_src.shape[-1]

    # cache epipolar line data (queries, log_base)
    xx, yy, mask = epipolar.lines_raster(epi_lines_tgt, res=r, dim=-2)  # 3 x (v, v, r, nl)
    xx[~mask], yy[~mask] = 0, 0
    line_data = torch.cat((query_imgs, img_rays, log_base), dim=-1)[
        torch.arange(v).view(1, v, 1, 1), yy, xx,
    ]  # (v, v, r, nl, e + 3 + 1)
    # set probability to zero, log(p) = -inf, outside raster mask
    line_data[~mask][..., -1] = -torch.inf
    # set probability to zero for own epipolar line
    line_data[
        torch.arange(v, device=device),
        torch.arange(v, device=device),
        ...,
        -1
    ] = -torch.inf

    # sample 2D pts from masks
    mask_prob_flat = torch.sigmoid(mask_lgts).view(v * r * r).cumsum_(dim=0)
    idx_mask = torch.searchsorted(mask_prob_flat, torch.rand(n_samples, device=device) * mask_prob_flat[-1])
    v_mask = torch.divide(idx_mask, r * r, rounding_mode='floor')
    xy_mask = idx_mask % (r * r)
    y_mask = torch.divide(xy_mask, r, rounding_mode='floor')
    x_mask = xy_mask % r

    if debug:
        vis = torch.zeros_like(mask_prob_flat)
        vis.scatter_add_(dim=0, index=idx_mask, src=torch.ones(n_samples, device=device))
        vis = (vis / vis.max()) ** 0.5
        vis = vis.view(v, r, r).permute(1, 0, 2).reshape(r, v * r)
        cv2.imshow('2d mask samples', vis.cpu().numpy())

    corr_obj = torch.empty(n_samples, n_2d_3d_corr, 1, 3, device=device)
    corr_world = torch.empty(n_samples, n_2d_3d_corr, n_epi_corr, 3, device=device)
    for i in range(0, n_samples, batch_size):
        vv, yy, xx = v_mask[i:i + batch_size], y_mask[i:i + batch_size], x_mask[i:i + batch_size]
        bs = len(vv)

        # get epipolar line data
        epi_idx = epi_lines_idx[vv, yy, xx]  # (bs, v)
        line_data_ = line_data[vv[:, None], torch.arange(v, device=device), :, epi_idx]  # (bs, v, r, e + 3 + 1)
        line_data_ = line_data_.view(bs, v * r, e + 3 + 1)
        # (bs, v*r, e), (bs, v*r, 3), (bs, v*r)
        line_q, line_rays, line_log_base = line_data_[..., :e], line_data_[..., e:e + 3], line_data_[..., -1]

        # compute 2D-3D correspondence
        prob_3d = torch.softmax((query_imgs[vv, yy, xx] / temp_2d_3d) @ obj_keys.T, dim=1).cumsum_(dim=1)  # (bs, m)
        idx_3d = torch.searchsorted(prob_3d, torch.rand(bs, n_2d_3d_corr, device=device) * prob_3d[:, -1:])

        corr_obj[i:i + bs] = obj_pts[idx_3d[..., None]]  # (bs, n_2d_3d_corr, 1, 3)
        k = obj_keys[idx_3d]  # (bs, n_2d_3d_corr, e)

        qk_dots = (k[:, :, None] * line_q[:, None]).sum(dim=-1)  # (bs, n_2d_3d_corr, v * r)
        epi_probs = qk_dots + line_log_base[:, None]
        if epi_view_prior_by_max:
            epi_probs = epi_probs.view(bs, n_2d_3d_corr, v, r)
            view_priors = torch.softmax(epi_probs.max(dim=-1, keepdim=True)[0], dim=-2)  # (vs, n_2d_3d_corr, v, 1)

        epi_probs = (epi_probs - epi_probs.max(dim=-1, keepdim=True)[0]).exp_()  # subtract max for numerical stability
        # TODO: Could epi_probs.max() be -inf? In that case, the sample should be dropped
        assert not epi_probs.isnan().any()

        if epi_uniform_view_priors:
            epi_probs = epi_probs.view(bs, n_2d_3d_corr, v, r)
            eps = torch.Tensor([1e-9], device=device)
            epi_probs = epi_probs / torch.maximum(epi_probs.mean(dim=-1, keepdim=True), eps)
            epi_probs = epi_probs.view(bs, n_2d_3d_corr, v * r)

        epi_probs = epi_probs.cumsum_(dim=-1)
        idx_epi_corr = torch.searchsorted(
            epi_probs,
            torch.rand(bs, n_2d_3d_corr, n_epi_corr, device=device) * epi_probs[..., -1:]
        )

        # Intersection between 2D mask sample ray and epi line sample ray
        ray_a = img_rays[vv, yy, xx]  # (bs, 3)
        center_a = world_p_cams[vv, :3, 0]  # (bs, 3)
        # (bs, n_2d_3d_corr, n_epi_corr, 3)
        ray_b = line_rays[torch.arange(bs, device=device)[:, None, None], idx_epi_corr]
        vv_b = torch.divide(idx_epi_corr, r, rounding_mode='floor')
        center_b = world_p_cams[vv_b, :3, 0]  # (bs, n_2d_3d_corr, n_epi_corr, 3)
        # (bs, n_2d_3d_corr, n_epi_corr, 3)
        corr_world[i:i + bs] = closest_point_to_lines(center_a[:, None, None], ray_a[:, None, None], center_b, ray_b)

        if i == 0 and debug:
            for k in range(bs):
                v_, y_, x_ = vv[k].item(), yy[k].item(), xx[k].item()
                ray_b_, vb_ = ray_b[k, 0, 0], vv_b[k, 0, 0].item()
                ray_b_ = cams_t_world[vb_, :3, :3] @ ray_b_
                ray_b_ = Ks[vb_] @ ray_b_
                xb_, yb_ = (ray_b_[:2] / ray_b_[2]).round().long().cpu().numpy().tolist()

                epi_idx_ = epi_idx[k]  # (v,)
                epi_probs_ = epi_probs[k, 0]  # cumsummed (v * r,)

                epi_probs_[1:] -= epi_probs_[:-1].clone()  # reverse cumsum
                epi_probs_ = (epi_probs_ / epi_probs_.max()) ** 0.5
                epi_probs_ = epi_probs_.view(1, v * r, 1).expand(20, v * r, 3)

                vis = query_imgs.view(v, r, r, 3, -1).mean(dim=-1)
                vis = vis / (2 * vis.abs().max()) + 0.5
                vis = vis.permute(1, 0, 2, 3).reshape(r, v * r, 3)

                vis_qline = line_q[k].view(v * r, 3, -1).mean(dim=-1)
                vis_qline = vis_qline / (2 * vis_qline.abs().max()) + 0.5
                vis_qline = vis_qline.view(1, v * r, 3).expand(20, v * r, 3)

                colors = torch.zeros(4, 3, device=device)
                colors[(0, 1, 2), (0, 1, 2)] = 1.
                for j, idx in enumerate(epi_idx_):
                    if j == v_:
                        continue
                    c = colors[j]
                    epipolar.draw_line(epi_lines_src[v_, j, :, idx], vis[:, r * v_:r * (v_ + 1)], c)
                    epipolar.draw_line(epi_lines_tgt[v_, j, :, idx], vis[:, r * j:r * (j + 1)], c)

                vis = torch.cat((vis, vis_qline, epi_probs_), dim=0)
                vis = vis.cpu().numpy()
                cv2.drawMarker(vis, (x_ + r * v_, y_), (0, 0, 0), cv2.MARKER_CROSS, 20)
                cv2.drawMarker(vis, (xb_ + r * vb_, yb_), (1, 1, 1), cv2.MARKER_CROSS, 20)

                cv2.imshow('epi prob lines', vis[..., ::-1])
                key = cv2.waitKey()
                if key == ord('q'):
                    break

    return corr_world, corr_obj


def sample_corr_triplets(corr_world: torch.tensor, corr_obj: torch.tensor,
                         sample_factor=1):
    r"""
    From 3D-3D correspondences, sample triplets where the triangle spanned by world coordinates is similar to the
    triangle spanned by object coordinates.
    :returns: corr (2[world,obj], m, 3corr, 3xyz), loss (m,) - sorted by loss
    """
    n_samples, n_2d_3d_corr, n_epi_corr, d = corr_world.shape
    assert d == 3 and corr_obj.shape == (n_samples, n_2d_3d_corr, 1, d), (corr_world.shape, corr_obj.shape)
    n = n_samples * n_2d_3d_corr * n_epi_corr
    corr = torch.stack(torch.broadcast_tensors(corr_world, corr_obj)).view(2, n, 3)
    m = n * sample_factor
    corr = corr[:, torch.randint(n, (m, 3), device=corr.device)]  # (2, m, 3corr, 3xyz)

    tri_side = corr[:, :, (1, 2, 2)] - corr[:, :, (0, 0, 1)]  # (2, m, 3(ab, ac, bc), 3xyz)
    tri_side_len = tri_side.norm(dim=-1)  # (2, m, 3)
    tri_side_len_diff = (tri_side_len[0] - tri_side_len[1]).norm(dim=-1)  # (m,)

    tri_area = torch.cross(tri_side[:, :, 0], tri_side[:, :, 1], dim=-1).norm(dim=-1)  # (2, m)
    min_tri_height = (tri_area / (tri_side_len.max(dim=-1)[0] + 1e-9)).min(dim=0)[0]  # (m,)
    loss = tri_side_len_diff / (min_tri_height + 1e-9)

    idx_sort = torch.argsort(loss)
    return corr[:, idx_sort], loss[idx_sort]


def kabsch(P: torch.tensor, Q: torch.tensor):
    """
    Q = R @ P + t
    """
    n, _, m = P.shape  # n sets of m 3D correspondences
    assert m >= 3
    assert P.shape == Q.shape == (n, 3, m), (P.shape, Q.shape)
    mp, mq = P.mean(dim=-1, keepdim=True), Q.mean(dim=-1, keepdim=True)
    P_, Q_ = P - mp, Q - mq
    C = Q_ @ P_.mT  # (n, 3, 3)
    u, _, v = torch.svd(C)  # (n, 3, 3), (n, 3), (n, 3, 3)
    s = torch.ones(n, 3, 1, device=P.device)
    s[:, 2, 0] = torch.det(u @ v.mT)
    R = u @ (s * v.mT)  # (n, 3, 3)
    t = mq - R @ mp  # (n, 3, 1)
    return R, t


if __name__ == '__main__':
    from scipy.spatial.transform import Rotation

    R = torch.from_numpy(Rotation.random().as_matrix()[None]).float()  # (1, 3, 3)

    n, m = 10, 3
    t = torch.randn(n, 3, 1)
    P = torch.randn(n, 3, m)
    Q = R @ P + t

    R_, t_ = kabsch(P, Q)
    print(torch.dist(R, R_))
    print(torch.dist(t, t_))
