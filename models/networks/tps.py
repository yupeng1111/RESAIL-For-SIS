import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy

EPSILON = 1e-10


def _cross_squared_distance_matrix(x, y):
    """
    Args:
        x: [batch_size, n, 2]
        y: [batch_size, m, 2]
    Returns:
        squared_distance: [batch_size, n, m]
        squared_distance[b, i, j] = ||x[b, i, :] - y[b, j, :]||^2
    """
    x_squared = torch.sum(torch.square(x), dim=-1)
    y_squared = torch.sum(torch.square(y), dim=-1)

    x_squared_tile = x_squared.unsqueeze(-1)
    y_squared_tile = y_squared.unsqueeze(-2)

    x_y_transpose = torch.matmul(x, y.transpose(2, 1))

    squared_distance = x_squared_tile.expand_as(x_y_transpose) - 2 * x_y_transpose + y_squared_tile.expand_as(
        x_y_transpose)
    return squared_distance


def _pairwise_squared_distance_matrix(x):
    """
    Args:
        x: [batch, n, 2]

    Returns:
        squared_distance: [batch_size, n, n]
        squared_distance[b, i, j] = ||x[b, i, :] - x[b, j,:]||^2
    """
    x = x.double()

    x_x_transpose = torch.matmul(x, x.transpose(2, 1))

    x_square = torch.diagonal(x_x_transpose, dim1=-2, dim2=-1)

    x_square_tile = x_square.unsqueeze(-1)

    squared_distance = x_square_tile - 2 * x_x_transpose + x_square_tile.transpose(2, 1)

    return squared_distance


def _phi(r):
    r = r.double()
    return 0.5 * r * torch.log(torch.where(r == 0, torch.ones_like(r) * EPSILON, r))


def _solve_interpolation(
        train_points,
        train_values,
        regularization_weight
):

    k = train_values.shape[2]
    b, n, d = train_points.shape
    c = train_points.double()
    f = train_values.double()

    A = _phi(_pairwise_squared_distance_matrix(c))

    if regularization_weight > 0:
        batch_identity_matrix = torch.eye(n).unsqueeze(0)
        A = A + regularization_weight * batch_identity_matrix  # broadcast
    ones = torch.ones([b, n, 1]).to(train_points.device)

    B = torch.cat([c, ones], dim=2)
    left_block = torch.cat([A, B.transpose(2, 1)], dim=1)
    num_b_cols = B.shape[2]
    lhs_zeros = torch.zeros([b, num_b_cols, num_b_cols]).to(train_points.device)
    right_block = torch.cat([B, lhs_zeros], dim=1)
    lhs = torch.cat([left_block, right_block], dim=2)
    rhs_zores = torch.zeros([b, d + 1, k]).to(train_points.device)
    rhs = torch.cat([f, rhs_zores], dim=1)

    w_v = torch.linalg.solve(lhs, rhs)
    # w_v = torch.matmul(lhs.inverse() , rhs)
    w = w_v[:, :n, :]
    v = w_v[:, n:, :]

    return w, v


def _apply_interpolation(query_points, train_points, w, v):
    batch_size = train_points.shape[0]
    num_query_points = query_points.shape[1]

    pairwise_distance = _cross_squared_distance_matrix(query_points, train_points)
    phi_pairwise_distance = _phi(pairwise_distance)

    rbf_term = torch.matmul(phi_pairwise_distance, w)

    query_points_pad = torch.cat(
        [query_points, torch.ones([batch_size, num_query_points, 1]).to(query_points.device)],
        dim=2
    )

    linear_term = torch.matmul(query_points_pad, v)
    return rbf_term + linear_term


def interpolate_spline(
        train_points,
        train_values,
        query_points,
        regularization_weight=0.
):

    w, v = _solve_interpolation(train_points, train_values, regularization_weight)

    query_values = _apply_interpolation(query_points, train_points, w, v)

    return query_values


def _get_boundary_locations(image_height, image_width, num_points_per_edge):
    """Compute evenly-spaced indices along edge of image."""
    y_range = torch.linspace(0, image_height - 1, num_points_per_edge + 2).double()
    x_range = torch.linspace(0, image_width - 1, num_points_per_edge + 2).double()
    ys, xs = torch.meshgrid(y_range, x_range)

    is_boundary = (xs == 0) | \
                  (xs == image_width - 1) | \
                  (ys == 0) | \
                  (ys == image_height - 1)
    return torch.stack([ys[is_boundary], xs[is_boundary]], dim=-1)


def _add_zero_flow_controls_at_boundary(control_point_locations,
                                        control_point_flows, image_height,
                                        image_width, boundary_points_per_edge):

    batch_size = control_point_locations.shape[0]

    boundary_point_locations = _get_boundary_locations(image_height, image_width,
                                                       boundary_points_per_edge)

    boundary_point_flows = torch.zeros([boundary_point_locations.shape[0], 2]).double()

    boundary_point_locations = torch.repeat_interleave(boundary_point_locations.unsqueeze(0), repeats=batch_size, dim=0)

    boundary_point_flows = torch.repeat_interleave(boundary_point_flows.unsqueeze(0), repeats=batch_size, dim=0)

    merged_control_point_locations = torch.cat(
        [control_point_locations, boundary_point_locations], dim=1)

    merged_control_point_flows = torch.cat(
        [control_point_flows, boundary_point_flows], dim=1)

    return merged_control_point_locations, merged_control_point_flows



def spares_image_warp(
        image,
        source_control_point_locations,
        dest_control_point_locations,
        regularization_weight=0.,
        num_boundary_points=0
):
    image_datatype = image.dtype
    dest_control_point_locations = dest_control_point_locations.double()
    source_control_point_locations = source_control_point_locations.double()
    image = image.double()

    control_point_flows = dest_control_point_locations - source_control_point_locations

    batch_size, _, image_height, image_width = image.shape

    clamp_boundaries = num_boundary_points > 0
    boundary_points_per_edge = num_boundary_points - 1

    if clamp_boundaries:
        (dest_control_point_locations,
         control_point_flows) = _add_zero_flow_controls_at_boundary(
            dest_control_point_locations, control_point_flows, image_height,
            image_width, boundary_points_per_edge)

    y_grid = torch.arange(image_height)
    x_grid = torch.arange(image_width)
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid)

    grid_locations = torch.stack([y_grid, x_grid], dim=-1).double().to(image.device)
    flattend_grid_locations = grid_locations.view(image_height * image_width, 2)

    flattened_grid_locations = torch.repeat_interleave(flattend_grid_locations.unsqueeze(0), dim=0, repeats=batch_size)

    flattened_flows = interpolate_spline(
        dest_control_point_locations, control_point_flows,
        flattened_grid_locations, regularization_weight)

    dense_flows = flattened_flows.view(batch_size, image_height, image_width, 2)

    dense_flows_ = torch.zeros_like(dense_flows)
    dense_flows_[..., 0] = (grid_locations[..., 1] - dense_flows[..., 1]) / image_width * 2 - 1
    dense_flows_[..., 1] = (grid_locations[..., 0] - dense_flows[..., 0]) / image_height * 2 - 1
    warped_image = F.grid_sample(image.double(), dense_flows_, align_corners=False)
    return warped_image.type(image_datatype), dense_flows_

