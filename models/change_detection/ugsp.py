"""Uncertainty-aware Geometric Significance Prior (UGSP).

UGSP is evaluated once on the input pair, from P0 to P1.  All neighbourhood
operations are sparse radius/KNN queries; no dense pairwise distance matrix is
constructed.
"""

import time
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn import knn
import torch_points_kernels as tp


class UGSP(nn.Module):
    """Compute a one-way continuous geometric significance prior on P0."""

    def __init__(
        self,
        radius: float = 2.5,
        max_neighbors: int = 48,
        max_candidates: int = 64,
        alpha: float = 0.5,
        graph_k: Optional[int] = None,
        rho: Optional[float] = None,
        h: Optional[float] = None,
        eta: float = 1.0,
        xi: float = 0.5,
        omega: float = 0.5,
        registration_error: float = 0.1,
        beta: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if radius <= 0:
            raise ValueError("UGSP radius must be positive")
        if max_neighbors < 3:
            raise ValueError("UGSP max_neighbors must be at least 3")
        if max_candidates < 1:
            raise ValueError("UGSP max_candidates must be positive")
        if beta <= 0:
            raise ValueError("UGSP beta must be positive")
        if registration_error < 0:
            raise ValueError("UGSP registration_error must be non-negative")

        self.radius = float(radius)
        self.max_neighbors = int(max_neighbors)
        self.max_candidates = int(max_candidates)
        self.alpha = float(alpha)
        self.graph_k = min(int(graph_k or max_neighbors), self.max_neighbors - 1)
        self.rho = float(rho if rho is not None else radius)
        self.h = float(h if h is not None else radius)
        self.eta = float(eta)
        self.xi = float(xi)
        self.omega = float(omega)
        self.registration_error = float(registration_error)
        self.beta = float(beta)
        self.eps = float(eps)
        self.last_details: Dict[str, torch.Tensor] = {}
        self.last_timings: Dict[str, float] = {}

    @torch.no_grad()
    def forward(
        self,
        pos0: torch.Tensor,
        pos1: torch.Tensor,
        batch0: Optional[torch.Tensor] = None,
        batch1: Optional[torch.Tensor] = None,
        collect_timings: bool = False,
    ) -> torch.Tensor:
        """Return the continuous P0 -> P1 prior with shape ``[N0, 1]``."""

        pos0 = self._as_positions(pos0).float()
        pos1 = self._as_positions(pos1).float()
        batch0 = self._as_batch(batch0, pos0)
        batch1 = self._as_batch(batch1, pos1)

        started = self._timestamp(pos0, collect_timings)
        geometry0 = self._local_geometry(pos0, batch0)
        geometry1 = self._local_geometry(pos1, batch1)
        geometry_time = self._elapsed(pos0, started, collect_timings)

        started = self._timestamp(pos0, collect_timings)
        match_idx = self._match_points(pos0, pos1, batch0, batch1, geometry0["normal"], geometry1["normal"])
        correspondence_time = self._elapsed(pos0, started, collect_timings)

        target_position = geometry1["position"][match_idx]
        target_normal = geometry1["normal"][match_idx]
        target_theta = geometry1["theta"][match_idx]
        target_q_l = geometry1["q_l"][match_idx]
        target_q_o = geometry1["q_o"][match_idx]

        delta = target_position - geometry0["position"]
        d_l = torch.abs(torch.sum(delta * geometry0["normal"], dim=-1))
        d_n = self._angle(geometry0["normal"], target_normal)
        d_theta = torch.abs(geometry0["theta"] - target_theta)
        d_o = self.omega * d_n + (1.0 - self.omega) * d_theta

        g_l = 1.96 * torch.sqrt(geometry0["q_l"].square() + target_q_l.square() + self.registration_error**2)
        g_o = 1.96 * torch.sqrt(geometry0["q_o"].square() + target_q_o.square())
        ratio_l = d_l / (g_l + self.eps)
        ratio_o = d_o / (g_o + self.eps)
        significance_ratio = torch.maximum(ratio_l, ratio_o)
        prior = torch.sigmoid(self.beta * (significance_ratio - 1.0))
        hard = (significance_ratio > 1.0).to(prior.dtype)

        self.last_details = {
            "hard": hard.unsqueeze(-1),
            "d_l": d_l,
            "g_l": g_l,
            "d_o": d_o,
            "g_o": g_o,
            "ratio_l": ratio_l,
            "ratio_o": ratio_o,
            "match_index": match_idx,
            "neighbor_count0": geometry0["neighbor_count"],
            "neighbor_count1": geometry1["neighbor_count"],
        }
        self.last_timings = {
            "geometry": geometry_time,
            "correspondence": correspondence_time,
        }
        return prior.unsqueeze(-1)

    def _local_geometry(self, pos: torch.Tensor, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        idx, valid = self._radius_indices(pos, pos, batch, batch, self.radius, self.max_neighbors)
        n_points = pos.shape[0]

        # A self support makes isolated points numerically well-defined.
        rows = torch.arange(n_points, device=pos.device)
        idx[:, 0] = rows
        valid[:, 0] = True
        idx_safe = idx.clamp_min(0)
        neighbours = pos[idx_safe]
        offsets = neighbours - pos.unsqueeze(1)
        distance = torch.linalg.norm(offsets, dim=-1)

        weights = valid.to(pos.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        weights = weights / count.unsqueeze(-1)
        covariance = torch.einsum("nk,nki,nkj->nij", weights, offsets, offsets)
        identity = torch.eye(3, device=pos.device, dtype=pos.dtype).unsqueeze(0)
        _, eigenvectors = torch.linalg.eigh(covariance + self.eps * identity)
        normal = _normalize(eigenvectors[:, :, 0], self.eps)

        offset_unit = offsets / distance.unsqueeze(-1).clamp_min(self.eps)
        cos_theta = torch.abs(torch.sum(offset_unit * normal.unsqueeze(1), dim=-1)).clamp(0.0, 1.0)
        theta_value = torch.abs(torch.acos(cos_theta) - torch.pi / 2.0)
        theta_valid = valid & (distance > self.eps)
        theta_count = theta_valid.to(pos.dtype).sum(dim=1).clamp_min(1.0)
        theta = (theta_value * theta_valid.to(pos.dtype)).sum(dim=1) / theta_count

        normal_offset = torch.sum(offsets * normal.unsqueeze(1), dim=-1)
        roughness = torch.sqrt((normal_offset.square() * weights).sum(dim=1).clamp_min(0.0))
        q_l = roughness / torch.sqrt(count)

        neighbour_normal = normal[idx_safe]
        neighbour_normal = self._align_normals(normal, neighbour_normal)
        normal_angle = self._angle(normal.unsqueeze(1), neighbour_normal)

        graph_valid = valid & (distance > self.eps)
        if graph_valid.shape[1] > self.graph_k + 1:
            graph_valid[:, self.graph_k + 1 :] = False
        no_edge = ~graph_valid.any(dim=1)
        graph_valid[no_edge, 0] = True
        graph_weights = self._graph_weights(distance, normal_angle, graph_valid, pos.dtype)

        first_position = (graph_weights.unsqueeze(-1) * neighbours).sum(dim=1)
        first_normal = _normalize((graph_weights.unsqueeze(-1) * neighbour_normal).sum(dim=1), self.eps)
        first_theta = (graph_weights * theta[idx_safe]).sum(dim=1)

        first_position_neighbour = first_position[idx_safe]
        first_normal_neighbour = self._align_normals(first_normal, first_normal[idx_safe])
        second_position = (graph_weights.unsqueeze(-1) * first_position_neighbour).sum(dim=1)
        second_normal = _normalize((graph_weights.unsqueeze(-1) * first_normal_neighbour).sum(dim=1), self.eps)
        second_theta = (graph_weights * first_theta[idx_safe]).sum(dim=1)

        norm = 1.0 + self.alpha + self.alpha**2
        position = (pos + self.alpha * first_position + self.alpha**2 * second_position) / norm
        first_normal = self._align_to_reference(normal, first_normal)
        second_normal = self._align_to_reference(normal, second_normal)
        propagated_normal = _normalize(normal + self.alpha * first_normal + self.alpha**2 * second_normal, self.eps)
        theta = (theta + self.alpha * first_theta + self.alpha**2 * second_theta) / norm

        normal_rotation = normal_angle / distance.clamp_min(self.eps)
        curvature = torch.sqrt((normal_rotation.square() * weights).sum(dim=1).clamp_min(0.0))
        q_o = self.radius * curvature / torch.sqrt(count)

        # Standard deviations propagate through weighted sums as sqrt(sum(w^2 q^2)).
        first_q_l = self._propagate_std(q_l, idx_safe, graph_weights)
        second_q_l = self._propagate_std(first_q_l, idx_safe, graph_weights)
        q_l = self._combine_std(q_l, first_q_l, second_q_l, norm)
        first_q_o = self._propagate_std(q_o, idx_safe, graph_weights)
        second_q_o = self._propagate_std(first_q_o, idx_safe, graph_weights)
        q_o = self._combine_std(q_o, first_q_o, second_q_o, norm)

        return {
            "position": position,
            "normal": propagated_normal,
            "theta": theta,
            "q_l": q_l,
            "q_o": q_o,
            "neighbor_count": count,
        }

    def _match_points(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_batch: torch.Tensor,
        target_batch: torch.Tensor,
        source_normal: torch.Tensor,
        target_normal: torch.Tensor,
    ) -> torch.Tensor:
        search_radius = (self.rho**2 + self.h**2) ** 0.5
        candidate, valid = self._radius_indices(
            target,
            source,
            target_batch,
            source_batch,
            search_radius,
            self.max_candidates,
        )
        candidate_safe = candidate.clamp_min(0)
        candidate_position = target[candidate_safe]
        candidate_normal = target_normal[candidate_safe]
        delta = candidate_position - source.unsqueeze(1)
        parallel_signed = torch.sum(delta * source_normal.unsqueeze(1), dim=-1)
        d_parallel = torch.abs(parallel_signed)
        d_perpendicular = torch.linalg.norm(delta - parallel_signed.unsqueeze(-1) * source_normal.unsqueeze(1), dim=-1)
        angle = self._angle(source_normal.unsqueeze(1), candidate_normal)
        feasible = valid & (d_perpendicular <= self.rho) & (d_parallel <= self.h)
        score = d_perpendicular + self.eta * d_parallel + self.xi * angle
        score = score.masked_fill(~feasible, float("inf"))
        best_score, best_slot = score.min(dim=1)
        best = candidate_safe.gather(1, best_slot.unsqueeze(1)).squeeze(1)

        # Changed or missing surfaces may have no local candidate.  A sparse
        # global 1-NN fallback keeps every source point defined without cdist.
        missing = ~torch.isfinite(best_score)
        if missing.any():
            nearest = self._nearest_indices(target, source, target_batch, source_batch)
            best[missing] = nearest[missing]
        return best

    @staticmethod
    def _radius_indices(
        support: torch.Tensor,
        query: torch.Tensor,
        support_batch: torch.Tensor,
        query_batch: torch.Tensor,
        radius: float,
        max_neighbors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx, squared_distance = tp.ball_query(
            radius,
            max_neighbors,
            support,
            query,
            mode="partial_dense",
            batch_x=support_batch,
            batch_y=query_batch,
            # The CUDA kernel does not implement sorted ball queries. Sort the
            # small sparse result below so graph_k still selects nearby points.
            sort=False,
        )
        idx = idx.long()
        order = squared_distance.masked_fill(idx < 0, float("inf")).argsort(dim=1)
        idx = idx.gather(1, order)
        return idx, idx >= 0

    @staticmethod
    def _nearest_indices(
        support: torch.Tensor,
        query: torch.Tensor,
        support_batch: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        edge = knn(support, query, 1, support_batch, query_batch)
        nearest = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        nearest[edge[0]] = edge[1]
        return nearest

    def _graph_weights(
        self,
        distance: torch.Tensor,
        normal_angle: torch.Tensor,
        valid: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        valid_float = valid.to(dtype)
        distance_weight = valid_float / (distance + self.eps)
        distance_weight = distance_weight / distance_weight.sum(dim=1, keepdim=True).clamp_min(self.eps)
        direction_weight = valid_float / (normal_angle + self.eps)
        direction_weight = direction_weight / direction_weight.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return 0.5 * (distance_weight + direction_weight)

    @staticmethod
    def _align_normals(center: torch.Tensor, neighbours: torch.Tensor) -> torch.Tensor:
        sign = torch.where(
            torch.sum(center.unsqueeze(1) * neighbours, dim=-1, keepdim=True) < 0,
            -torch.ones((), dtype=neighbours.dtype, device=neighbours.device),
            torch.ones((), dtype=neighbours.dtype, device=neighbours.device),
        )
        return neighbours * sign

    @staticmethod
    def _align_to_reference(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        sign = torch.where(
            torch.sum(reference * value, dim=-1, keepdim=True) < 0,
            -torch.ones((), dtype=value.dtype, device=value.device),
            torch.ones((), dtype=value.dtype, device=value.device),
        )
        return value * sign

    @staticmethod
    def _propagate_std(q: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((weights.square() * q[idx].square()).sum(dim=1).clamp_min(0.0))

    def _combine_std(
        self,
        base: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        norm: float,
    ) -> torch.Tensor:
        variance = base.square() + self.alpha**2 * first.square() + self.alpha**4 * second.square()
        return torch.sqrt(variance.clamp_min(0.0)) / norm

    @staticmethod
    def _angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        cosine = torch.abs(torch.sum(first * second, dim=-1)).clamp(0.0, 1.0)
        return torch.acos(cosine)

    @staticmethod
    def _as_positions(pos: torch.Tensor) -> torch.Tensor:
        if pos is None or pos.dim() != 2 or pos.shape[1] != 3:
            raise ValueError("UGSP positions must have shape [N, 3]")
        if pos.shape[0] == 0:
            raise ValueError("UGSP requires non-empty point clouds")
        return pos

    @staticmethod
    def _as_batch(batch: Optional[torch.Tensor], pos: torch.Tensor) -> torch.Tensor:
        if batch is None:
            return torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        batch = batch.to(device=pos.device, dtype=torch.long).reshape(-1)
        if batch.shape[0] != pos.shape[0]:
            raise ValueError("UGSP batch vectors must match the point count")
        return batch

    @staticmethod
    def _timestamp(reference: torch.Tensor, enabled: bool) -> Optional[float]:
        if not enabled:
            return None
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        return time.perf_counter()

    @staticmethod
    def _elapsed(reference: torch.Tensor, started: Optional[float], enabled: bool) -> float:
        if not enabled or started is None:
            return 0.0
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        return time.perf_counter() - started


def _normalize(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.linalg.norm(value, dim=-1, keepdim=True).clamp_min(eps)
