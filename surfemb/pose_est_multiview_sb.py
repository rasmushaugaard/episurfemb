import torch

from surfemb.utils import timer

device = torch.device('cuda:0')

v = 4
e = 12
r = 224
n = r * r
m = 70_000

s = 5_000
s_des = 100_000

queries = torch.randn(v, n, e, device=device)
mask_probs = torch.softmax(torch.randn(v, n, device=device), dim=1)
mask_probs = mask_probs.view(-1).cumsum(dim=0)
keys = torch.rand(m, e, device=device)
log_denominator = torch.logsumexp(queries @ keys[:1000].T, dim=-1).view(v, r, r).permute(1, 0, 2).reshape(r, v * r)

for _ in range(3):
    print()
    with timer('total'):
        with timer('sample mask'):  # vn + s_des * log2(vn) ~ 4 * 224^2 + 1e5 * log2(4*224^2) ~ 2e5 + 1e5 * 17 ~ 1e6
            # sample from mask probs
            idx_2d = torch.searchsorted(mask_probs, torch.rand(s, device=device) * mask_probs[-1])
            view_idx = torch.div(idx_2d, n, rounding_mode='floor')
            pixel_idx = idx_2d % n
            pixel_idx.view(-1)[0].item()
        with timer('sample 3d'):  # s_des * m * e ~ 1e5 * 1e4 * 1e1 = 1e10
            # sampling 3D is def. the most limiting part, both wrt time and space
            # Alternatively, simply sample 3D points randomly and find two 2D corr. in different views.
            # Then the probability of a 3D point being visible is of interest. For objects with continuous symmetries,
            # this will be very high though, and objects without symmetries are a lot easier anyway.
            # Wait. -_-. Then we would need (s_des * n ~ 1e5 * 5e4 = 5e9) to sample the first 3D-2D corr. singleview,
            # and 3D-2D is prob not as effective as 2D-3D

            # But, we could sample n_3d per distribution instead of just 1, almost for free. n3d * log2()
            qk_probs = torch.softmax(queries[view_idx, pixel_idx] @ keys.T, dim=1).cumsum_(dim=1)  # (s, m)
            idx_3d = torch.searchsorted(qk_probs, torch.rand(s, 1, device=device) * qk_probs[:, -1:])[:, 0]  # (s,)
            del qk_probs
            k = keys[idx_3d]  # (s, e)
            k.view(-1)[0].item()

        with timer('sample epi'):  # s_des * e + s_des * v * r + s_des ~ 1e5 * 1e3 = 1e8
            # just take random lines for now
            line_idx = torch.randint(r, (s,), device=device)
            query_lines = queries.view(v, r, r, e)[:, line_idx].permute(1, 0, 2, 3).reshape(s, v * r, e)  # reshape pri
            log_nominator = (query_lines * k.view(s, 1, e)).sum(dim=-1)  # (s, v * r)
            log_denom = log_denominator[line_idx]  # (s, v * r)
            prob = (log_nominator - log_denom).exp_().cumsum_(dim=1)  # (s, v * r)  # TODO: reweigh
            # in case of a continuous symmetry, an epipolar line that is parallel to that symmetry does not provide much
            # disambiguation, but it would constitute much of the probability mass across views, which is not desired.
            # One way to alleviate this could be to normalize within views, such that all other views have equal priors.
            idx_2d_other = torch.searchsorted(prob, torch.rand(s, 1, device=device) * prob[:, -1:])[:, 0]  # (s,)
            idx_2d_other[-1].item()

        with timer('solve 3d intersection by least squares'):
            l0, l1 = torch.randn(2, s, 4, device=device)
            A = torch.stack((l0[:, :3], l1[:, :3]), dim=1)  # (s, 2, 3)
            b = torch.stack((l0[:, 3], l1[:, 3]), dim=1)  # (s, 2)
            p = torch.linalg.lstsq(A, b)[0]
            p.view(-1)[0].item()

        with timer('solve 3d intersection by 2D cross product'):
            l0, l1 = torch.randn(2, s, 3, device=device)
            p = torch.cross(l0, l1)
            p.view(-1)[0].item()

        with timer('solve 3d intersection 3 eq in 3 unknowns'):
            # Approx 20 x times faster than using least squares on 3D-lines.
            # Could probably be faster with epipolar plane cashing and calculating intersection as 2D cross product,
            # but it would arguably be more complex, and this is already fast enough to not be the limiting factor.

            c0, c1, x0, x1 = torch.randn(4, s, 3, device=device)
            x2 = torch.cross(x0, x1)
            # c0 + t0x0 = c1 + t1x1 + t2x2
            # t0x0 - t1x1 - t2x2 = c1 - c0
            A = torch.stack((x0, -x1, -x2), dim=-1)  # (s, 3eq(xyz), 3unknowns)
            b = c1 - c0  # (s, 3eq(xyz))
            t = torch.linalg.solve(A, b)  # (s, 3unknowns)
            p = c0 + t[:, :1] * x0

            p.view(-1)[0].item()
