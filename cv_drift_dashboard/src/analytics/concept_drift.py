class PageHinkley:
    """Page-Hinkley test for concept drift detection on streaming metrics."""
    def __init__(self, min_instances=30, delta=0.005, threshold=50, alpha=0.9999):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = 0.0
        self.n = 0
        self.sum = 0.0
        
    def add_element(self, x):
        self.n += 1
        self.x_mean = self.x_mean + (x - self.x_mean) / self.n
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)
        
        if self.n < self.min_instances:
            return False
            
        return self.sum > self.threshold

def calculate_concept_drift(historical_accuracies: list, new_accuracy: float, ph_detector=None):
    """
    Calculate concept drift based on model performance metrics (requires ground truth).
    Tracks error rate instead of accuracy because PH detects increases in the mean.
    """
    if ph_detector is None:
        ph_detector = PageHinkley()
        for acc in historical_accuracies:
            ph_detector.add_element(1.0 - acc)
            
    # Add new element
    error_rate = 1.0 - new_accuracy
    is_drift = ph_detector.add_element(error_rate)
    
    # Score proxy: how close is the sum to the threshold
    score = min(max(ph_detector.sum / ph_detector.threshold, 0.0), 1.0) if ph_detector.threshold > 0 else 0.0
    
    return {
        "score": float(score),
        "is_drift": bool(is_drift),
        "current_sum": float(ph_detector.sum),
        "details": f"PH Sum: {ph_detector.sum:.2f} (Threshold: {ph_detector.threshold})"
    }
