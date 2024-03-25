import cv2
import torch


def point_on_lines(lines: torch.Tensor, line_axis=-1):
    assert lines.shape[line_axis] == 3
    lines = torch.swapaxes(lines, 0, line_axis)
    a, b, c = lines
    mask_x = abs(a) > abs(b)
    mask_y = ~mask_x
    pts = torch.zeros_like(lines)
    pts[2] = 1.
    pts[0, mask_x] = -c[mask_x] / a[mask_x]
    pts[1, mask_y] = -c[mask_y] / b[mask_y]
    return torch.swapaxes(pts, 0, line_axis)


def lines_raster(lines: torch.Tensor, res: int, dim=-1):
    dim = dim % lines.ndim
    assert lines.shape[dim] == 3
    a, b, c = lines.split(1, dim)
    mask_x = abs(a) > abs(b)
    mask_y = ~mask_x

    # bring new dimension to the right to enable broadcasting
    x = torch.zeros((*a.shape, res), dtype=torch.long, device=lines.device)
    y = x.clone()
    rng = torch.arange(res, device=lines.device)
    a, b, c = [t.unsqueeze(-1) for t in (a, b, c)]

    y[mask_x] = rng
    x[mask_x] = torch.round(- (c[mask_x] + b[mask_x] * rng) / a[mask_x]).long()

    x[mask_y] = rng
    y[mask_y] = torch.round(- (c[mask_y] + a[mask_y] * rng) / b[mask_y]).long()

    x, y = x.swapaxes(dim, -1).squeeze(-1), y.swapaxes(dim, -1).squeeze(-1)
    mask = (0 <= x) & (x < res) & (0 <= y) & (y < res)
    return x, y, mask


def draw_line(l: torch.Tensor, im: torch.Tensor, color: torch.Tensor):
    h, w = im.shape[:2]
    assert h == w
    x, y, mask = lines_raster(l, res=h)
    im[y[mask], x[mask]] = color


def get_fundamental_matrices(P: torch.Tensor, world_p_cams: torch.Tensor):
    # https://sourishghosh.com/2016/fundamental-matrix-from-camera-matrices/
    v = len(P)
    assert P.shape == (v, 3, 4), P.shape
    assert world_p_cams.shape == (v, 4, 1), world_p_cams.shape
    eye = torch.eye(3, device=P.device)
    P_pinv = torch.pinverse(P)
    F = torch.cross((P[None] @ world_p_cams[:, None]).mT, -eye, dim=-1) @ P[None] @ P_pinv[:, None]
    assert F.shape == (v, v, 3, 3)
    return F


def get_boundary_indices(res: int, device: torch.device):
    ones = torch.ones(res, device=device)
    idx = torch.arange(res, device=device)
    lt = torch.zeros(res, device=device)
    br = lt + (res - 1)
    # (x, y, 1)
    p_boundary = torch.cat((
        torch.stack((idx, lt, ones)),  # top
        torch.stack((idx, br, ones)),  # bottom
        torch.stack((lt, idx, ones))[:, 1:-1],  # left
        torch.stack((br, idx, ones))[:, 1:-1],  # right
    ), dim=1)  # (3, 4 * (crop_res - 1))
    assert p_boundary.shape == (3, 4 * (res - 1))
    return p_boundary


def build_epipolar_line_index(F: torch.Tensor, res: int):
    # storing all epipolar line queries for all pts becomes prohibitive (v * n * v * r * e ~ 2e9)
    # Each epipolar line is shared among many pixels in the source image, so instead
    # compute lines only for boundary pixels (v * 4r * v * r * e ~ 40e6)
    # and then store an index to the lines (v * n * v ~ 1e6).
    v = len(F)
    assert F.shape == (v, v, 3, 3)
    # find epipolar lines in the other (target) images
    epi_lines_t = F @ get_boundary_indices(res, device=F.device)  # (v, v, 3, nl)
    nl = epi_lines_t.shape[-1]
    # find the corresponding epipolar lines in the source image
    epi_line_pts_t = point_on_lines(epi_lines_t, line_axis=-2)
    epi_lines_s = torch.swapaxes(F, 0, 1) @ epi_line_pts_t  # (v, v, 3, nl)

    # for each line in the source, store the index of that line
    epi_line_index = torch.empty((v, res, res, v), device=F.device, dtype=torch.long)  # (s, sy, sx, t)
    epi_line_index.fill_(-1)
    epi_line_index[torch.arange(v, device=F.device), :, :, torch.arange(v, device=F.device)] = 0
    x, y, mask = lines_raster(epi_lines_s, res=res, dim=-2)  # 3 x (n_views_s, n_views_t, r, nl)
    x, y, mask, vs, vt, epi_idx = torch.broadcast_tensors(
        x, y, mask,
        torch.arange(v, device=F.device).view((v, 1, 1, 1)),
        torch.arange(v, device=F.device).view((1, v, 1, 1)),
        torch.arange(nl, device=F.device).view((1, 1, 1, nl)),
    )
    epi_line_index[vs[mask], y[mask], x[mask], vt[mask]] = epi_idx[mask]
    empty_mask = epi_line_index == -1
    if empty_mask.any():
        print('WARNING: empty epipolar cache index:', empty_mask.float().sum() / (v * (v - 1) * res * res))
    return epi_line_index, epi_lines_s, epi_lines_t
