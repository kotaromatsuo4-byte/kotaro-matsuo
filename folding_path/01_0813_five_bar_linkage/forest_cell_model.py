"""Dependency-free terminal-cell classifiers and a conservative duplicate filter.

The five-bar search produces one expensive label per completed perturbation
macro::

    (parent coordinates, path context, first displacement, K) -> terminal cell

The labels from residual suffixes of the same macro are useful, but they are
not independent observations.  :class:`TerminalCellClassifier` therefore
bootstraps *macro groups*, gives every group the same total training weight,
and reports the number of independent groups rather than the number of rows.

This module intentionally implements only a one-sided decision: reject a
candidate when there is strong, in-support evidence that it returns to a
known cell.  Uncertain and out-of-distribution candidates are retained.  A
single observed class can never support such a classification decision, so
the filter is disabled until at least two classes have been observed.

Scikit-learn is not a project dependency.  The small NumPy implementation
below provides bagged, axis-aligned random-forest and extra-trees variants.
It is designed for the modest online datasets used by the folding examples,
not as a general replacement for a production tree library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DuplicateFilterThresholds:
    """Conservative thresholds for known-duplicate rejection.

    These defaults are deliberately strict.  They must eventually be tuned
    by chronological replay and leave-one-cell-out pseudo-unknown validation,
    rather than by ordinary classification accuracy alone.
    """

    mean_probability: float = 0.90
    lower_probability: float = 0.70
    vote_fraction: float = 0.80
    max_disagreement: float = 0.08
    max_support_ratio: float = 1.00
    min_groups_per_class: int = 4


def make_macro_feature(
    node: np.ndarray,
    context_node: np.ndarray,
    displacement: np.ndarray,
    k: int,
    *,
    length_scale: float,
    free_indices: Optional[Sequence[int]] = None,
    reference_point: float = 0.0,
    k_max: int = 24,
) -> np.ndarray:
    """Encode ``(q, context, physical displacement, K)`` without SVD modes.

    ``node`` and ``context_node`` use the same convention as ``main8.py``:
    the reference height is removed from z and coordinates are normalized by
    a characteristic member length.  Translation in x/y is expected to have
    already been fixed by the structural boundary conditions.
    """

    if not np.isfinite(length_scale) or float(length_scale) <= 0.0:
        raise ValueError("length_scale must be positive and finite")
    if int(k) <= 0:
        raise ValueError("k must be positive")
    if int(k_max) <= 0:
        raise ValueError("k_max must be positive")

    q = np.asarray(node, dtype=float).reshape(-1, 3).copy()
    context = np.asarray(context_node, dtype=float).reshape(-1, 3).copy()
    if q.shape != context.shape:
        raise ValueError("node and context_node must have the same shape")
    v = np.asarray(displacement, dtype=float).reshape(-1)
    if v.size != q.size:
        raise ValueError("displacement must have one value per nodal coordinate")

    q[:, 2] -= float(reference_point)
    context[:, 2] -= float(reference_point)
    q_flat = q.reshape(-1)
    context_flat = context.reshape(-1)
    if free_indices is None:
        free = np.arange(q_flat.size, dtype=int)
    else:
        free = np.asarray(free_indices, dtype=int).reshape(-1)
        if np.any(free < 0) or np.any(free >= q_flat.size):
            raise ValueError("free_indices contains an out-of-range coordinate")

    scale = float(length_scale)
    log_k = np.log1p(float(k)) / np.log1p(float(k_max))
    feature = np.concatenate(
        (q_flat[free] / scale, context_flat[free] / scale, v[free] / scale, [log_k])
    )
    if not np.all(np.isfinite(feature)):
        raise ValueError("macro feature contains non-finite values")
    return feature


def permute_reject_mask(
    reject_mask: np.ndarray,
    random_state: Optional[int | np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute a reject mask while preserving its rejection count exactly."""

    mask = np.asarray(reject_mask, dtype=bool).reshape(-1)
    rng = _as_generator(random_state)
    permutation = rng.permutation(len(mask))
    return mask[permutation], permutation


def _as_generator(
    random_state: Optional[int | np.random.Generator],
) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _label_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = tuple(value.reshape(-1).tolist())
    elif isinstance(value, list):
        value = tuple(value)
    try:
        hash(value)
    except TypeError as error:
        raise TypeError("cell labels must be hashable scalar or tuple values") from error
    return value


def _object_array(values: Sequence[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result


def _encode_stable(values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    classes: list[Any] = []
    lookup: dict[Any, int] = {}
    codes = []
    for raw_value in values:
        value = _label_key(raw_value)
        if value not in lookup:
            lookup[value] = len(classes)
            classes.append(value)
        codes.append(lookup[value])
    return _object_array(classes), np.asarray(codes, dtype=int)


def _weighted_gini(class_weight: np.ndarray) -> float:
    total = float(np.sum(class_weight))
    if total <= 0.0:
        return 0.0
    probabilities = class_weight / total
    return float(1.0 - probabilities @ probabilities)


@dataclass
class _TreeNode:
    probabilities: np.ndarray
    feature: int = -1
    threshold: float = np.nan
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0


class _RandomizedClassificationTree:
    def __init__(
        self,
        *,
        num_classes: int,
        kind: str,
        max_depth: int,
        min_samples_leaf: int,
        max_features: str | int | float,
        max_thresholds: int,
        extra_thresholds_per_feature: int,
        leaf_smoothing: float,
        rng: np.random.Generator,
    ) -> None:
        self.num_classes = int(num_classes)
        self.kind = str(kind)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_thresholds = int(max_thresholds)
        self.extra_thresholds_per_feature = int(extra_thresholds_per_feature)
        self.leaf_smoothing = float(leaf_smoothing)
        self.rng = rng
        self.root: Optional[_TreeNode] = None

    def _feature_count(self, dimension: int) -> int:
        value = self.max_features
        if isinstance(value, str):
            if value == "sqrt":
                return max(1, int(np.sqrt(dimension)))
            if value == "log2":
                return max(1, int(np.log2(max(dimension, 2))))
            if value == "all":
                return dimension
            raise ValueError(f"unknown max_features value: {value}")
        if isinstance(value, (float, np.floating)):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("floating max_features must be in (0, 1]")
            return max(1, int(np.ceil(float(value) * dimension)))
        return min(dimension, max(1, int(value)))

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        indices = np.flatnonzero(np.asarray(weights, dtype=float) > 0.0)
        if len(indices) == 0:
            raise ValueError("a bootstrap tree received no positive-weight rows")
        self.root = self._grow(x, y, weights, indices, depth=0)

    def _leaf(self, y: np.ndarray, weights: np.ndarray, indices: np.ndarray) -> _TreeNode:
        counts = np.bincount(
            y[indices], weights=weights[indices], minlength=self.num_classes
        ).astype(float)
        counts += self.leaf_smoothing
        return _TreeNode(probabilities=counts / np.sum(counts))

    def _thresholds(self, values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            return np.empty(0, dtype=float)
        low = float(np.min(finite))
        high = float(np.max(finite))
        if not high > low:
            return np.empty(0, dtype=float)
        if self.kind == "extra_trees":
            return self.rng.uniform(
                low, high, size=max(1, self.extra_thresholds_per_feature)
            )

        unique = np.unique(finite)
        if len(unique) < 2:
            return np.empty(0, dtype=float)
        candidates = 0.5 * (unique[:-1] + unique[1:])
        if len(candidates) <= self.max_thresholds:
            return candidates
        positions = np.linspace(
            0, len(candidates) - 1, self.max_thresholds, dtype=int
        )
        return candidates[np.unique(positions)]

    def _grow(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> _TreeNode:
        leaf = self._leaf(y, weights, indices)
        positive_classes = np.unique(y[indices][weights[indices] > 0.0])
        if (
            depth >= self.max_depth
            or len(indices) < 2 * self.min_samples_leaf
            or len(positive_classes) <= 1
        ):
            return leaf

        dimension = x.shape[1]
        feature_count = self._feature_count(dimension)
        features = self.rng.choice(dimension, size=feature_count, replace=False)
        total_weight = float(np.sum(weights[indices]))
        parent_counts = np.bincount(
            y[indices], weights=weights[indices], minlength=self.num_classes
        )
        parent_impurity = _weighted_gini(parent_counts)
        best_gain = 0.0
        best_feature = -1
        best_threshold = np.nan
        best_left: Optional[np.ndarray] = None
        best_right: Optional[np.ndarray] = None

        for feature in features:
            values = x[indices, feature]
            for threshold in self._thresholds(values):
                left_mask = values <= threshold
                left_count = int(np.count_nonzero(left_mask))
                right_count = len(indices) - left_count
                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue
                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]
                left_weight = float(np.sum(weights[left_indices]))
                right_weight = float(np.sum(weights[right_indices]))
                if left_weight <= 0.0 or right_weight <= 0.0:
                    continue
                left_classes = np.bincount(
                    y[left_indices],
                    weights=weights[left_indices],
                    minlength=self.num_classes,
                )
                right_classes = parent_counts - left_classes
                child_impurity = (
                    left_weight * _weighted_gini(left_classes)
                    + right_weight * _weighted_gini(right_classes)
                ) / total_weight
                gain = parent_impurity - child_impurity
                if gain > best_gain + 1.0e-15:
                    best_gain = float(gain)
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left = left_indices
                    best_right = right_indices

        if best_feature < 0 or best_left is None or best_right is None:
            return leaf
        leaf.feature = best_feature
        leaf.threshold = best_threshold
        leaf.left = self._grow(x, y, weights, best_left, depth + 1)
        leaf.right = self._grow(x, y, weights, best_right, depth + 1)
        return leaf

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("tree is not fitted")
        result = np.empty((len(x), self.num_classes), dtype=float)
        for row_index, row in enumerate(x):
            node = self.root
            while not node.is_leaf:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
                if node is None:
                    raise RuntimeError("malformed classification tree")
            result[row_index] = node.probabilities
        return result


class TerminalCellClassifier:
    """Bagged terminal-cell classifier with a conservative filter API."""

    def __init__(
        self,
        *,
        model_kind: str = "random_forest",
        n_estimators: int = 64,
        max_depth: int = 10,
        min_samples_leaf: int = 2,
        max_features: str | int | float = "sqrt",
        max_thresholds: int = 32,
        extra_thresholds_per_feature: int = 2,
        leaf_smoothing: float = 0.5,
        support_neighbors: int = 3,
        support_quantile: float = 0.95,
        lower_probability_quantile: float = 0.10,
        balance_classes: bool = True,
        standardize: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if model_kind not in ("random_forest", "extra_trees"):
            raise ValueError("model_kind must be 'random_forest' or 'extra_trees'")
        if n_estimators <= 0 or max_depth <= 0 or min_samples_leaf <= 0:
            raise ValueError("tree counts, depth, and leaf size must be positive")
        if not 0.0 <= leaf_smoothing:
            raise ValueError("leaf_smoothing must be nonnegative")
        if support_neighbors <= 0:
            raise ValueError("support_neighbors must be positive")
        if not 0.0 < support_quantile <= 1.0:
            raise ValueError("support_quantile must be in (0, 1]")
        if not 0.0 <= lower_probability_quantile <= 0.5:
            raise ValueError("lower_probability_quantile must be in [0, .5]")

        self.model_kind = model_kind
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_thresholds = int(max_thresholds)
        self.extra_thresholds_per_feature = int(extra_thresholds_per_feature)
        self.leaf_smoothing = float(leaf_smoothing)
        self.support_neighbors = int(support_neighbors)
        self.support_quantile = float(support_quantile)
        self.lower_probability_quantile = float(lower_probability_quantile)
        self.balance_classes = bool(balance_classes)
        self.standardize = bool(standardize)
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.trees_: list[_RandomizedClassificationTree] = []
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_scale_: Optional[np.ndarray] = None
        self.training_x_: Optional[np.ndarray] = None
        self.training_y_: Optional[np.ndarray] = None
        self.training_groups_: Optional[np.ndarray] = None
        self.class_group_counts_: Optional[np.ndarray] = None
        self.support_radii_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.classes_ is not None

    @property
    def filter_ready(self) -> bool:
        return bool(self.is_fitted and len(self.classes_) >= 2)

    def _validate_x(self, inputs: np.ndarray, *, fitting: bool) -> np.ndarray:
        x = np.asarray(inputs, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError("inputs must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(x)):
            raise ValueError("inputs contain non-finite values")
        if not fitting and self.feature_mean_ is not None:
            if x.shape[1] != len(self.feature_mean_):
                raise ValueError("input feature dimension differs from fitted data")
        return x

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("classifier is not fitted")
        return (x - self.feature_mean_) / self.feature_scale_

    def fit(
        self,
        inputs: np.ndarray,
        labels: Sequence[Any],
        group_ids: Optional[Sequence[Any]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "TerminalCellClassifier":
        x_raw = self._validate_x(inputs, fitting=True)
        if len(labels) != len(x_raw):
            raise ValueError("labels must have one entry per input row")
        classes, y = _encode_stable(labels)
        if len(classes) == 0:
            raise ValueError("at least one training row is required")

        if group_ids is None:
            group_codes = np.arange(len(x_raw), dtype=int)
        else:
            if len(group_ids) != len(x_raw):
                raise ValueError("group_ids must have one entry per input row")
            _, group_codes = _encode_stable(group_ids)

        for group in np.unique(group_codes):
            if len(np.unique(y[group_codes == group])) != 1:
                raise ValueError("all residual rows in a macro group need one cell label")

        if sample_weight is None:
            row_weight = np.ones(len(x_raw), dtype=float)
        else:
            row_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(row_weight) != len(x_raw):
                raise ValueError("sample_weight must have one entry per input row")
            if np.any(~np.isfinite(row_weight)) or np.any(row_weight < 0.0):
                raise ValueError("sample weights must be finite and nonnegative")
        if not np.any(row_weight > 0.0):
            raise ValueError("at least one sample weight must be positive")

        # Normalize every independent macro to unit total weight.
        base_weight = row_weight.copy()
        unique_groups = np.unique(group_codes)
        for group in unique_groups:
            mask = group_codes == group
            total = float(np.sum(base_weight[mask]))
            if total > 0.0:
                base_weight[mask] /= total
            else:
                base_weight[mask] = 0.0

        if self.balance_classes and len(classes) > 1:
            totals = np.bincount(y, weights=base_weight, minlength=len(classes))
            positive = totals > 0.0
            factors = np.ones(len(classes), dtype=float)
            factors[positive] = np.mean(totals[positive]) / totals[positive]
            base_weight *= factors[y]

        if self.standardize:
            mean = np.average(x_raw, axis=0, weights=np.maximum(base_weight, 1.0e-15))
            centered = x_raw - mean
            variance = np.average(
                centered * centered,
                axis=0,
                weights=np.maximum(base_weight, 1.0e-15),
            )
            scale = np.sqrt(np.maximum(variance, 1.0e-12))
        else:
            mean = np.zeros(x_raw.shape[1], dtype=float)
            scale = np.ones(x_raw.shape[1], dtype=float)
        x = (x_raw - mean) / scale

        self.classes_ = classes
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        self.training_x_ = x
        self.training_y_ = y
        self.training_groups_ = group_codes
        self.class_group_counts_ = np.asarray(
            [len(np.unique(group_codes[y == cell])) for cell in range(len(classes))],
            dtype=int,
        )
        self.support_radii_ = self._fit_support_radii()

        rng = _as_generator(self.random_state)
        self.trees_ = []
        group_count = int(np.max(group_codes)) + 1
        for _ in range(self.n_estimators):
            sampled = rng.integers(0, group_count, size=group_count)
            multiplicity = np.bincount(sampled, minlength=group_count).astype(float)
            tree_weight = base_weight * multiplicity[group_codes]
            if not np.any(tree_weight > 0.0):
                continue
            tree = _RandomizedClassificationTree(
                num_classes=len(classes),
                kind=self.model_kind,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_thresholds=self.max_thresholds,
                extra_thresholds_per_feature=self.extra_thresholds_per_feature,
                leaf_smoothing=self.leaf_smoothing,
                rng=rng,
            )
            tree.fit(x, y, tree_weight)
            self.trees_.append(tree)
        if not self.trees_:
            raise RuntimeError("no bootstrap classification tree was fitted")
        return self

    @staticmethod
    def _rms_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if len(reference) == 0:
            return np.empty(0, dtype=float)
        difference = reference - query.reshape(1, -1)
        return np.sqrt(np.mean(difference * difference, axis=1))

    def _fit_support_radii(self) -> np.ndarray:
        if self.training_x_ is None or self.training_y_ is None:
            raise RuntimeError("training rows are not available")
        radii = np.full(len(self.classes_), np.nan, dtype=float)
        for cell in range(len(self.classes_)):
            mask = self.training_y_ == cell
            x_cell = self.training_x_[mask]
            groups = self.training_groups_[mask]
            unique_groups = np.unique(groups)
            if len(unique_groups) < 2:
                continue
            group_distances = []
            for group in unique_groups:
                own = x_cell[groups == group]
                other_groups = unique_groups[unique_groups != group]
                distances = []
                for other_group in other_groups:
                    other = x_cell[groups == other_group]
                    pair_difference = own[:, None, :] - other[None, :, :]
                    pair_distance = np.sqrt(
                        np.mean(pair_difference * pair_difference, axis=2)
                    )
                    distances.append(float(np.min(pair_distance)))
                distances.sort()
                neighbor_index = min(self.support_neighbors, len(distances)) - 1
                group_distances.append(distances[neighbor_index])
            radius = float(np.quantile(group_distances, self.support_quantile))
            radii[cell] = max(radius, 1.0e-12)
        return radii

    def _committee_probabilities(self, inputs: np.ndarray) -> np.ndarray:
        if not self.is_fitted or not self.trees_:
            raise RuntimeError("classifier is not fitted")
        x_raw = self._validate_x(inputs, fitting=False)
        x = self._transform(x_raw)
        return np.asarray([tree.predict_proba(x) for tree in self.trees_])

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """Return mean known-class probabilities, shape ``(n, classes)``."""

        return np.mean(self._committee_probabilities(inputs), axis=0)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(inputs)
        indices = np.argmax(probabilities, axis=1)
        return _object_array([self.classes_[index] for index in indices])

    def _support_distances(self, x: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        result = np.full(len(x), np.inf, dtype=float)
        for row_index, (row, cell) in enumerate(zip(x, predicted)):
            mask = self.training_y_ == int(cell)
            references = self.training_x_[mask]
            groups = self.training_groups_[mask]
            group_distances = []
            for group in np.unique(groups):
                distances = self._rms_distances(row, references[groups == group])
                group_distances.append(float(np.min(distances)))
            if not group_distances:
                continue
            group_distances.sort()
            neighbor_index = min(self.support_neighbors, len(group_distances)) - 1
            result[row_index] = group_distances[neighbor_index]
        return result

    def predict_diagnostics(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Return probabilities, committee uncertainty, and support diagnostics."""

        committee = self._committee_probabilities(inputs)
        mean = np.mean(committee, axis=0)
        predicted = np.argmax(mean, axis=1)
        row_indices = np.arange(len(mean))
        selected_probabilities = committee[:, row_indices, predicted]
        lower = np.quantile(
            selected_probabilities, self.lower_probability_quantile, axis=0
        )
        votes = np.argmax(committee, axis=2)
        vote_fraction = np.mean(votes == predicted.reshape(1, -1), axis=0)

        epsilon = 1.0e-15
        mean_entropy = -np.sum(mean * np.log(np.maximum(mean, epsilon)), axis=1)
        tree_entropy = -np.sum(
            committee * np.log(np.maximum(committee, epsilon)), axis=2
        )
        if mean.shape[1] > 1:
            disagreement = (mean_entropy - np.mean(tree_entropy, axis=0)) / np.log(
                mean.shape[1]
            )
            normalized_entropy = mean_entropy / np.log(mean.shape[1])
        else:
            disagreement = np.zeros(len(mean), dtype=float)
            normalized_entropy = np.zeros(len(mean), dtype=float)

        x = self._transform(self._validate_x(inputs, fitting=False))
        support_distance = self._support_distances(x, predicted)
        radius = self.support_radii_[predicted]
        support_ratio = np.divide(
            support_distance,
            radius,
            out=np.full(len(mean), np.inf, dtype=float),
            where=np.isfinite(radius) & (radius > 0.0),
        )
        predicted_labels = _object_array(
            [self.classes_[index] for index in predicted]
        )
        return {
            "probabilities": mean,
            "predicted_class_indices": predicted,
            "predicted_labels": predicted_labels,
            "mean_probability": mean[row_indices, predicted],
            "lower_probability": lower,
            "vote_fraction": vote_fraction,
            "disagreement": np.maximum(disagreement, 0.0),
            "entropy": normalized_entropy,
            "support_distance": support_distance,
            "support_radius": radius,
            "support_ratio": support_ratio,
            "class_group_count": self.class_group_counts_[predicted],
        }

    def reject_known_duplicates(
        self,
        inputs: np.ndarray,
        known_cells: Optional[Iterable[Any]] = None,
        *,
        thresholds: Optional[DuplicateFilterThresholds] = None,
        permute: bool = False,
        random_state: Optional[int | np.random.Generator] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return a mask that rejects only high-confidence known duplicates.

        ``permute=True`` is the score-permutation control: it permutes the final
        mask, so it preserves the exact number of rejected candidates while
        destroying the association between model evidence and candidates.
        """

        diagnostics = self.predict_diagnostics(inputs)
        reject = np.zeros(len(diagnostics["mean_probability"]), dtype=bool)
        if not self.filter_ready:
            diagnostics["unpermuted_reject_mask"] = reject.copy()
            diagnostics["score_permutation"] = np.arange(len(reject), dtype=int)
            return reject, diagnostics

        prediction = _prediction_api_dict(self, diagnostics)
        reject = conservative_duplicate_mask(
            prediction,
            self.classes_ if known_cells is None else known_cells,
            thresholds=thresholds,
        )
        diagnostics["unpermuted_reject_mask"] = reject.copy()
        if permute:
            reject, permutation = permute_reject_mask(reject, random_state)
        else:
            permutation = np.arange(len(reject), dtype=int)
        diagnostics["score_permutation"] = permutation
        return reject, diagnostics


def _coerce_thresholds(
    thresholds: Optional[DuplicateFilterThresholds | dict[str, Any]],
) -> DuplicateFilterThresholds:
    if thresholds is None:
        return DuplicateFilterThresholds()
    if isinstance(thresholds, DuplicateFilterThresholds):
        return thresholds
    if isinstance(thresholds, dict):
        return DuplicateFilterThresholds(**thresholds)
    raise TypeError("thresholds must be DuplicateFilterThresholds, dict, or None")


def _prediction_api_dict(
    model: TerminalCellClassifier,
    diagnostics: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Add stable, main-loop-facing names to internal diagnostics."""

    supported = (
        np.isfinite(diagnostics["support_ratio"])
        & (diagnostics["support_ratio"] <= 1.0)
    )
    result = dict(diagnostics)
    result.update(
        {
            "classes": model.classes_.copy(),
            "max_probability": diagnostics["mean_probability"],
            "committee_agreement": diagnostics["vote_fraction"],
            "epistemic": diagnostics["disagreement"],
            "nearest_class_distance": diagnostics["support_distance"],
            "supported": supported,
        }
    )
    return result


def fit_cell_filter(
    inputs: np.ndarray,
    labels: Sequence[Any],
    group_ids: Optional[Sequence[Any]] = None,
    sample_weight: Optional[Sequence[float]] = None,
    *,
    model_kind: str = "random_forest",
    rng_seed: Optional[int] = None,
    **model_options: Any,
) -> TerminalCellClassifier:
    """Fit and return a group-balanced terminal-cell committee.

    This functional wrapper is the stable integration API for main programs;
    the class remains public for offline diagnostics and advanced use.
    """

    if "random_state" in model_options:
        if rng_seed is not None:
            raise ValueError("pass either rng_seed or random_state, not both")
        rng_seed = model_options.pop("random_state")
    model = TerminalCellClassifier(
        model_kind=model_kind,
        random_state=rng_seed,
        **model_options,
    )
    return model.fit(inputs, labels, group_ids, sample_weight)


def predict_cell_filter(
    model: TerminalCellClassifier,
    query_inputs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Predict known cells and diagnostics without making a reject decision."""

    if not isinstance(model, TerminalCellClassifier):
        raise TypeError("model must be a TerminalCellClassifier")
    return _prediction_api_dict(model, model.predict_diagnostics(query_inputs))


def conservative_duplicate_mask(
    prediction: dict[str, np.ndarray],
    rejectable_classes: Iterable[Any],
    thresholds: Optional[DuplicateFilterThresholds | dict[str, Any]] = None,
) -> np.ndarray:
    """Make the one-sided filter decision from a prediction dictionary.

    The operation is deliberately independent of the fitted object so main
    loops can log or permute predictions before applying the same policy.
    """

    limits = _coerce_thresholds(thresholds)
    classes = np.asarray(prediction["classes"], dtype=object).reshape(-1)
    count = len(np.asarray(prediction["max_probability"]).reshape(-1))
    if len(classes) < 2:
        return np.zeros(count, dtype=bool)
    known = {_label_key(value) for value in rejectable_classes}
    labels = np.asarray(prediction["predicted_labels"], dtype=object).reshape(-1)
    is_known = np.asarray([_label_key(value) in known for value in labels], dtype=bool)
    return (
        is_known
        & (np.asarray(prediction["max_probability"]) >= limits.mean_probability)
        & (
            np.asarray(prediction["lower_probability"])
            >= limits.lower_probability
        )
        & (
            np.asarray(prediction["committee_agreement"])
            >= limits.vote_fraction
        )
        & (np.asarray(prediction["epistemic"]) <= limits.max_disagreement)
        & (
            np.asarray(prediction["support_ratio"])
            <= limits.max_support_ratio
        )
        & (
            np.asarray(prediction["class_group_count"])
            >= limits.min_groups_per_class
        )
    )


__all__ = [
    "conservative_duplicate_mask",
    "DuplicateFilterThresholds",
    "fit_cell_filter",
    "TerminalCellClassifier",
    "make_macro_feature",
    "permute_reject_mask",
    "predict_cell_filter",
]
