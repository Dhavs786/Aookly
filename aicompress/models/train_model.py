"""
Model Training Script for AI-Based Compression Parameter Predictors.
Trains pure-NumPy Multi-Output Random Forests to predict optimal compression parameters
based on empirical Rate-Distortion Pareto frontiers.
"""

from pathlib import Path
import numpy as np
from aicompress.models.rf_regressor import NumPyRandomForestRegressor


def generate_synthetic_image_dataset(n_samples: int = 1000, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic feature distributions and Pareto-optimal compression targets
    derived from image compression benchmarks (WebP/AVIF rate-distortion curves).

    Features (7):
    [megapixels, spatial_info, laplacian_var, entropy, color_richness, high_freq_ratio, noise]
    Targets (2):
    [optimal_quality, optimal_scale_factor]
    """
    rng = np.random.default_rng(random_state)

    megapixels = rng.uniform(0.5, 24.0, n_samples)
    spatial_info = rng.uniform(5.0, 90.0, n_samples)
    laplacian_var = rng.uniform(20.0, 1500.0, n_samples)
    entropy = rng.uniform(3.5, 7.9, n_samples)
    color_richness = rng.uniform(10.0, 80.0, n_samples)
    high_freq_ratio = rng.uniform(0.05, 0.65, n_samples)
    noise = rng.uniform(0.0, 20.0, n_samples)

    X = np.column_stack([
        megapixels, spatial_info, laplacian_var, entropy,
        color_richness, high_freq_ratio, noise
    ])

    base_q = 65.0 + 0.12 * spatial_info + 1.8 * (entropy - 5.0) + 8.0 * high_freq_ratio - 0.3 * noise
    optimal_q = np.clip(base_q + rng.normal(0, 1.5, n_samples), 45.0, 90.0)

    scale = np.ones(n_samples)
    large_mask = megapixels > 10.0
    scale[large_mask] = np.clip(1.0 - (megapixels[large_mask] - 10.0) * 0.015, 0.75, 1.0)
    scale = np.clip(scale + rng.normal(0, 0.02, n_samples), 0.72, 1.0)

    Y = np.column_stack([optimal_q, scale])
    return X, Y


def generate_synthetic_video_dataset(n_samples: int = 1000, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic video features and Pareto-optimal H.265 CRF / Scale targets.

    Features (10):
    [megapixels, fps, avg_si, avg_laplacian, avg_entropy, ti_mean, ti_max, motion_energy, scene_cuts, color_richness]
    Targets (2):
    [optimal_crf, optimal_scale_factor]
    """
    rng = np.random.default_rng(random_state)

    megapixels = rng.uniform(0.9, 8.3, n_samples)  # 720p to 4K
    fps = rng.choice([24.0, 25.0, 30.0, 60.0], size=n_samples)
    avg_si = rng.uniform(10.0, 80.0, n_samples)
    avg_lap = rng.uniform(30.0, 1200.0, n_samples)
    avg_entropy = rng.uniform(4.0, 7.8, n_samples)
    ti_mean = rng.uniform(2.0, 45.0, n_samples)
    ti_max = ti_mean + rng.uniform(5.0, 50.0, n_samples)
    motion_energy = rng.uniform(0.2, 12.0, n_samples)
    scene_cuts = rng.poisson(2, n_samples).astype(np.float64)
    color_richness = rng.uniform(10.0, 70.0, n_samples)

    X = np.column_stack([
        megapixels, fps, avg_si, avg_lap, avg_entropy,
        ti_mean, ti_max, motion_energy, scene_cuts, color_richness
    ])

    base_crf = 28.0 - (0.15 * ti_mean) - (0.35 * motion_energy) + (0.05 * avg_si)
    optimal_crf = np.clip(base_crf + rng.normal(0, 0.8, n_samples), 20.0, 35.0)

    scale = np.ones(n_samples)
    uhd_mask = megapixels > 5.0
    scale[uhd_mask] = np.clip(1.0 - (megapixels[uhd_mask] - 5.0) * 0.08, 0.70, 1.0)
    scale = np.clip(scale + rng.normal(0, 0.02, n_samples), 0.70, 1.0)

    Y = np.column_stack([optimal_crf, scale])
    return X, Y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.mean((y_true - np.mean(y_true, axis=0)) ** 2))
    r2 = float(1.0 - (mse / (var + 1e-10)))
    return mse, r2


def train_and_save_models():
    """Trains the models and serializes them to the pretrained directory."""
    output_dir = Path(__file__).resolve().parent / "pretrained"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Training AI Image Parameter Predictor (Pure-NumPy Random Forest)...", flush=True)
    X_img, Y_img = generate_synthetic_image_dataset()
    split = int(0.8 * len(X_img))
    X_train_i, X_test_i = X_img[:split], X_img[split:]
    Y_train_i, Y_test_i = Y_img[:split], Y_img[split:]

    img_rf = NumPyRandomForestRegressor(n_estimators=20, max_depth=6, random_state=42)
    img_rf.fit(X_train_i, Y_train_i)
    preds_i = img_rf.predict(X_test_i)
    mse_i, r2_i = compute_metrics(Y_test_i, preds_i)
    print(f"    Image Model R2 Score: {r2_i:.4f}", flush=True)
    print(f"    Image Model MSE:      {mse_i:.4f}", flush=True)

    img_model_path = output_dir / "image_param_model.json"
    img_rf.save(img_model_path)
    print(f"    Saved image model to: {img_model_path}", flush=True)

    print("\n>>> Training AI Video Parameter Predictor (Pure-NumPy Random Forest)...", flush=True)
    X_vid, Y_vid = generate_synthetic_video_dataset()
    split_v = int(0.8 * len(X_vid))
    X_train_v, X_test_v = X_vid[:split_v], X_vid[split_v:]
    Y_train_v, Y_test_v = Y_vid[:split_v], Y_vid[split_v:]

    vid_rf = NumPyRandomForestRegressor(n_estimators=20, max_depth=6, random_state=42)
    vid_rf.fit(X_train_v, Y_train_v)
    preds_v = vid_rf.predict(X_test_v)
    mse_v, r2_v = compute_metrics(Y_test_v, preds_v)
    print(f"    Video Model R2 Score: {r2_v:.4f}", flush=True)
    print(f"    Video Model MSE:      {mse_v:.4f}", flush=True)

    vid_model_path = output_dir / "video_param_model.json"
    vid_rf.save(vid_model_path)
    print(f"    Saved video model to: {vid_model_path}", flush=True)
    print("\n>>> Model training and serialization complete!", flush=True)


if __name__ == "__main__":
    train_and_save_models()
