import numpy as np
from scipy.stats import entropy

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KL divergence between two probability distributions."""
    # Add small epsilon to avoid division by zero or log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(np.sum(entropy(p, q)))

def calculate_prediction_drift(baseline_preds: np.ndarray, prod_preds: np.ndarray, threshold=0.1):
    """
    Calculate prediction drift based on prediction probabilities.
    baseline_preds and prod_preds should be arrays of shape (N, num_classes)
    """
    if len(baseline_preds) == 0 or len(prod_preds) == 0:
        return {"score": 0.0, "is_drift": False, "details": "Insufficient data"}
        
    p_baseline = np.mean(baseline_preds, axis=0)
    p_prod = np.mean(prod_preds, axis=0)
    
    kl_div = calculate_kl_divergence(p_baseline, p_prod)
    
    is_drift = kl_div > threshold
    
    # Normalize score somewhat for dashboard (max at 1.0)
    score = min(kl_div / (threshold * 2), 1.0) if threshold > 0 else 0.0
    
    return {
        "score": float(score),
        "is_drift": bool(is_drift),
        "kl_divergence": float(kl_div),
        "details": f"KL Divergence: {kl_div:.4f}"
    }
