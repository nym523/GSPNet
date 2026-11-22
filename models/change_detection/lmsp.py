import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch_geometric.nn import knn


class LMSP(nn.Module):
    """Latent multi-modal statistical prior based on online diagonal GMMs.

    The GMM parameters are buffers rather than trainable parameters: training
    batches update them with EM and an exponential moving average, while eval
    forwards only evaluate the persisted distributions.
    """

    def __init__(
        self,
        feature_dim: int,
        num_components: int = 3,
        em_iterations: int = 5,
        momentum: float = 0.1,
        min_samples: int = 8,
        variance_floor: float = 1e-3,
        unchanged_labels: Sequence[int] = (0,),
        ignore_label: Optional[int] = -1,
        class_counts: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("LMSP feature_dim must be positive")
        if num_components <= 0:
            raise ValueError("LMSP num_components must be positive")
        if em_iterations <= 0:
            raise ValueError("LMSP em_iterations must be positive")
        if not 0.0 < momentum <= 1.0:
            raise ValueError("LMSP momentum must be in (0, 1]")
        if min_samples <= 0:
            raise ValueError("LMSP min_samples must be positive")
        if variance_floor <= 0.0:
            raise ValueError("LMSP variance_floor must be positive")

        self.feature_dim = int(feature_dim)
        self.descriptor_dim = 2 * self.feature_dim + 1
        self.num_components = int(num_components)
        self.em_iterations = int(em_iterations)
        self.momentum = float(momentum)
        self.min_samples = int(min_samples)
        self.variance_floor = float(variance_floor)
        self.unchanged_labels = tuple(int(label) for label in unchanged_labels)
        self.ignore_label = ignore_label
        self.eps = float(eps)

        self.register_buffer("mixture_weights", torch.full((2, self.num_components), 1.0 / self.num_components))
        self.register_buffer("means", torch.zeros((2, self.num_components, self.descriptor_dim)))
        self.register_buffer("variances", torch.ones((2, self.num_components, self.descriptor_dim)))
        self.register_buffer("initialized", torch.zeros(2, dtype=torch.bool))
        warmup_capacity = max(self.min_samples, self.num_components)
        self.register_buffer(
            "_warmup_samples",
            torch.zeros((2, warmup_capacity, self.descriptor_dim)),
            persistent=False,
        )
        self.register_buffer("_warmup_count", torch.zeros(2, dtype=torch.long), persistent=False)

        if class_counts is None:
            parent_counts = torch.ones(2, dtype=torch.float32)
            self._fixed_class_prior = False
        else:
            class_counts = torch.as_tensor(class_counts, dtype=torch.float32).reshape(-1)
            unchanged = class_counts.new_zeros(())
            for label in self.unchanged_labels:
                if 0 <= label < class_counts.numel():
                    unchanged = unchanged + class_counts[label]
            parent_counts = torch.stack((unchanged, class_counts.sum() - unchanged)).clamp_min(0.0)
            if parent_counts.sum() <= 0:
                parent_counts = torch.ones(2, dtype=torch.float32)
            self._fixed_class_prior = True
        self.register_buffer("class_mass", parent_counts)

        self.last_details: Dict[str, object] = {}

    @property
    def ready(self) -> bool:
        return bool(self.initialized.all().item())

    def forward(
        self,
        features0: torch.Tensor,
        features1: torch.Tensor,
        pos0: torch.Tensor,
        pos1: torch.Tensor,
        batch0: Optional[torch.Tensor] = None,
        batch1: Optional[torch.Tensor] = None,
        labels0: Optional[torch.Tensor] = None,
        labels1: Optional[torch.Tensor] = None,
        update: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_features(features0, features1)
        batch0 = self._as_batch(batch0, pos0)
        batch1 = self._as_batch(batch1, pos1)
        match01 = self.nearest_indices(pos1, pos0, batch1, batch0)
        match10 = self.nearest_indices(pos0, pos1, batch0, batch1)

        descriptor0 = self.change_descriptor(features0, features1[match01])
        descriptor1 = self.change_descriptor(features1, features0[match10])
        should_update = self.training if update is None else bool(update)

        with torch.no_grad():
            descriptor0_fp32 = torch.nan_to_num(descriptor0.detach().float())
            descriptor1_fp32 = torch.nan_to_num(descriptor1.detach().float())
            if should_update and labels0 is not None and labels1 is not None:
                descriptors = torch.cat((descriptor0_fp32, descriptor1_fp32), dim=0)
                labels = torch.cat((labels0.reshape(-1), labels1.reshape(-1)), dim=0)
                self.update_statistics(descriptors, labels)

            if self.ready:
                prior0 = self.posterior_changed(descriptor0_fp32)
                prior1 = self.posterior_changed(descriptor1_fp32)
            else:
                prior0 = descriptor0_fp32.new_full((descriptor0.shape[0],), 0.5)
                prior1 = descriptor1_fp32.new_full((descriptor1.shape[0],), 0.5)

        self.last_details = {
            "ready": self.ready,
            "match01": match01,
            "match10": match10,
            "class_prior": self.class_prior.detach().clone(),
        }
        return prior0.to(features0.dtype).unsqueeze(-1), prior1.to(features1.dtype).unsqueeze(-1)

    @staticmethod
    def change_descriptor(source: torch.Tensor, matched_target: torch.Tensor) -> torch.Tensor:
        difference_norm = torch.linalg.norm(source - matched_target, dim=-1, keepdim=True)
        return torch.cat((source, matched_target, difference_norm), dim=-1)

    @property
    def class_prior(self) -> torch.Tensor:
        mass = self.class_mass.clamp_min(self.eps)
        return mass / mass.sum()

    def update_statistics(self, descriptors: torch.Tensor, labels: torch.Tensor) -> None:
        if descriptors.dim() != 2 or descriptors.shape[1] != self.descriptor_dim:
            raise ValueError("LMSP descriptors have an invalid shape")
        if labels.numel() != descriptors.shape[0]:
            raise ValueError("LMSP labels must match the descriptor count")

        labels = labels.to(device=descriptors.device, dtype=torch.long).reshape(-1)
        valid = torch.ones_like(labels, dtype=torch.bool)
        if self.ignore_label is not None:
            valid &= labels != int(self.ignore_label)
        valid &= labels >= 0
        if not valid.any():
            return

        binary = self._binary_labels(labels)
        valid_binary = binary[valid]
        if not self._fixed_class_prior:
            counts = torch.bincount(valid_binary, minlength=2).to(self.class_mass.dtype)
            if self.class_mass.sum() <= 2.0 and not self.initialized.any():
                self.class_mass.copy_(counts.clamp_min(1.0))
            else:
                self.class_mass.lerp_(counts.clamp_min(1.0), self.momentum)

        for parent in range(2):
            samples = descriptors[valid & (binary == parent)]
            required = max(self.min_samples, self.num_components)
            was_initialized = bool(self.initialized[parent].item())
            if not was_initialized:
                buffered_count = int(self._warmup_count[parent].item())
                if buffered_count:
                    samples = torch.cat((self._warmup_samples[parent, :buffered_count], samples), dim=0)
                if samples.shape[0] < required:
                    self._warmup_samples[parent, : samples.shape[0]].copy_(samples)
                    self._warmup_count[parent] = samples.shape[0]
                    continue
                self._warmup_count[parent] = 0
            elif samples.shape[0] < required:
                continue

            if was_initialized:
                weights = self.mixture_weights[parent]
                means = self.means[parent]
                variances = self.variances[parent]
            else:
                weights, means, variances = self._initial_parameters(samples)

            weights, means, variances = self._em(samples, weights, means, variances)
            if was_initialized:
                self.mixture_weights[parent].lerp_(weights, self.momentum)
                self.means[parent].lerp_(means, self.momentum)
                self.variances[parent].lerp_(variances, self.momentum)
            else:
                self.mixture_weights[parent].copy_(weights)
                self.means[parent].copy_(means)
                self.variances[parent].copy_(variances)
                self.initialized[parent] = True

            self.mixture_weights[parent].div_(self.mixture_weights[parent].sum().clamp_min(self.eps))
            self.variances[parent].clamp_(min=self.variance_floor)

    def posterior_changed(self, descriptors: torch.Tensor) -> torch.Tensor:
        if not self.ready:
            return descriptors.new_full((descriptors.shape[0],), 0.5)
        parent_log_prob = []
        log_prior = torch.log(self.class_prior.clamp_min(self.eps))
        for parent in range(2):
            component_log_prob = self._component_log_prob(
                descriptors,
                self.mixture_weights[parent],
                self.means[parent],
                self.variances[parent],
            )
            parent_log_prob.append(torch.logsumexp(component_log_prob, dim=1) + log_prior[parent])
        return torch.softmax(torch.stack(parent_log_prob, dim=1), dim=1)[:, 1]

    def _em(
        self,
        samples: torch.Tensor,
        weights: torch.Tensor,
        means: torch.Tensor,
        variances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for _ in range(self.em_iterations):
            log_prob = self._component_log_prob(samples, weights, means, variances)
            responsibilities = torch.softmax(log_prob, dim=1)
            component_mass = responsibilities.sum(dim=0).clamp_min(self.eps)
            weights = component_mass / component_mass.sum()
            means = responsibilities.transpose(0, 1).matmul(samples) / component_mass.unsqueeze(1)
            second_moment = responsibilities.transpose(0, 1).matmul(samples.square())
            second_moment = second_moment / component_mass.unsqueeze(1)
            variances = (second_moment - means.square()).clamp_min(self.variance_floor)
        return weights, means, variances

    def _initial_parameters(
        self, samples: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = torch.argsort(torch.linalg.norm(samples, dim=1))
        slots = torch.linspace(0, samples.shape[0] - 1, self.num_components, device=samples.device).long()
        means = samples[order[slots]].clone()
        variance = samples.var(dim=0, unbiased=False).clamp_min(self.variance_floor)
        variances = variance.unsqueeze(0).expand(self.num_components, -1).clone()
        weights = samples.new_full((self.num_components,), 1.0 / self.num_components)
        return weights, means, variances

    def _component_log_prob(
        self,
        samples: torch.Tensor,
        weights: torch.Tensor,
        means: torch.Tensor,
        variances: torch.Tensor,
    ) -> torch.Tensor:
        variances = variances.clamp_min(self.variance_floor)
        difference = samples.unsqueeze(1) - means.unsqueeze(0)
        mahalanobis = (difference.square() / variances.unsqueeze(0)).sum(dim=-1)
        log_determinant = torch.log(variances).sum(dim=-1)
        normalizer = self.descriptor_dim * math.log(2.0 * math.pi)
        return -0.5 * (mahalanobis + log_determinant.unsqueeze(0) + normalizer) + torch.log(
            weights.clamp_min(self.eps)
        ).unsqueeze(0)

    def _binary_labels(self, labels: torch.Tensor) -> torch.Tensor:
        unchanged = torch.zeros_like(labels, dtype=torch.bool)
        for label in self.unchanged_labels:
            unchanged |= labels == label
        return (~unchanged).long()

    def _validate_features(self, features0: torch.Tensor, features1: torch.Tensor) -> None:
        for features in (features0, features1):
            if features.dim() != 2 or features.shape[1] != self.feature_dim:
                raise ValueError("LMSP features must have shape [N, feature_dim]")
            if features.shape[0] == 0:
                raise ValueError("LMSP requires non-empty feature tensors")

    @staticmethod
    def _as_batch(batch: Optional[torch.Tensor], pos: torch.Tensor) -> torch.Tensor:
        if batch is None:
            return torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        batch = batch.to(device=pos.device, dtype=torch.long).reshape(-1)
        if batch.shape[0] != pos.shape[0]:
            raise ValueError("LMSP batch vectors must match their point counts")
        return batch

    @staticmethod
    def nearest_indices(
        support: torch.Tensor,
        query: torch.Tensor,
        support_batch: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        edge = knn(support, query, 1, support_batch, query_batch)
        if edge.shape[1] != query.shape[0]:
            raise ValueError("LMSP could not find one cross-temporal match for every query point")
        nearest = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
        nearest[edge[0]] = edge[1]
        return nearest
