"""
Pure-NumPy Random Forest Multi-Output Regressor.
Enables fast, robust machine learning without external C-extension/DLL dependencies.
"""

from typing import List, Optional, Tuple, Dict, Any
import json
from pathlib import Path
import numpy as np


class TreeNode:
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[np.ndarray] = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf vector output

    def is_leaf(self) -> bool:
        return self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        if self.is_leaf():
            return {"value": self.value.tolist() if isinstance(self.value, np.ndarray) else [float(v) for v in self.value]}
        return {
            "feature": int(self.feature) if self.feature is not None else None,
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TreeNode":
        if "value" in d:
            return cls(value=np.array(d["value"], dtype=np.float32))
        return cls(
            feature=d["feature"],
            threshold=d["threshold"],
            left=cls.from_dict(d["left"]) if d.get("left") else None,
            right=cls.from_dict(d["right"]) if d.get("right") else None,
        )


class NumPyDecisionTreeRegressor:
    """Fast Multi-Output Decision Tree Regressor in pure NumPy."""

    def __init__(self, max_depth: int = 7, min_samples_split: int = 4, max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        self.root = self._build_tree(X, y, depth=0, rng=rng)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator) -> TreeNode:
        n_samples, n_features = X.shape

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=np.mean(y, axis=0))

        # Sample feature subset
        n_feats_to_try = self.max_features if self.max_features else max(1, int(np.sqrt(n_features)))
        feature_indices = rng.choice(n_features, size=n_feats_to_try, replace=False)

        best_feat = None
        best_thresh = None
        best_mse = float("inf")
        best_splits = None

        current_mse = np.sum(np.var(y, axis=0))

        for feat in feature_indices:
            col = X[:, feat]
            # Test percentiles for fast threshold evaluation
            percentiles = np.percentile(col, np.linspace(10, 90, 8))
            for thresh in percentiles:
                left_mask = col <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue

                mse_left = np.sum(np.var(y[left_mask], axis=0)) * np.sum(left_mask)
                mse_right = np.sum(np.var(y[right_mask], axis=0)) * np.sum(right_mask)
                total_mse = (mse_left + mse_right) / n_samples

                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feat = feat
                    best_thresh = float(thresh)
                    best_splits = (left_mask, right_mask)

        # If no significant improvement, return leaf
        if best_splits is None or (current_mse - best_mse) < 1e-4:
            return TreeNode(value=np.mean(y, axis=0))

        left_mask, right_mask = best_splits
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)

        return TreeNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child
        )

    def predict_single(self, x: np.ndarray, node: TreeNode) -> np.ndarray:
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        return self.predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_single(x, self.root) for x in X], dtype=np.float32)


class NumPyRandomForestRegressor:
    """Ensemble of Multi-Output Decision Trees in pure NumPy."""

    def __init__(self, n_estimators: int = 25, max_depth: int = 6, min_samples_split: int = 4, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees: List[NumPyDecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = NumPyDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max(1, int(np.sqrt(X.shape[1]) * 1.5))
            )
            tree.fit(X[boot_idx], y[boot_idx], rng=rng)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Model has not been fitted.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

    def save(self, file_path: Path | str):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            "trees": [tree.root.to_dict() for tree in self.trees if tree.root]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, file_path: Path | str) -> "NumPyRandomForestRegressor":
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rf = cls(
            n_estimators=data["n_estimators"],
            max_depth=data["max_depth"],
            min_samples_split=data["min_samples_split"],
            random_state=data["random_state"]
        )
        rf.trees = []
        for tree_dict in data["trees"]:
            t = NumPyDecisionTreeRegressor(max_depth=rf.max_depth, min_samples_split=rf.min_samples_split)
            t.root = TreeNode.from_dict(tree_dict)
            rf.trees.append(t)
        return rf
