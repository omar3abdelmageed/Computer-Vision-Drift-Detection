import torch
import torchvision.models as models
import torchvision.transforms as transforms
from alibi_detect.cd import MMDDrift
import numpy as np

# Singleton model loading to save memory
_resnet = None
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_resnet_extractor():
    global _resnet
    if _resnet is None:
        # Load a pretrained ResNet18 and remove the final classification layer
        resnet = models.resnet18(pretrained=True)
        _resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        _resnet.eval()
    return _resnet

def extract_embeddings(images_np: list) -> np.ndarray:
    """Takes a list of numpy arrays (H,W,C) and returns embeddings."""
    if not images_np:
        return np.array([])
        
    extractor = get_resnet_extractor()
    embeddings = []
    with torch.no_grad():
        for img in images_np:
            # Ensure img is uint8 for ToPILImage
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            tensor = _preprocess(img).unsqueeze(0)
            emb = extractor(tensor)
            embeddings.append(emb.squeeze().numpy())
    return np.array(embeddings)

def calculate_data_drift(baseline_images: list, production_images: list, p_val_threshold=0.05):
    """Calculates data drift using Maximum Mean Discrepancy (MMD) on ResNet embeddings."""
    if not baseline_images or not production_images:
        return {"score": 0.0, "is_drift": False, "details": "Insufficient data"}

    x_ref = extract_embeddings(baseline_images)
    x_test = extract_embeddings(production_images)
    
    # Initialize MMD drift detector
    # backend='pytorch' is used by default if PyTorch is installed, 
    # but we can omit it to let Alibi decide or explicitly specify.
    cd = MMDDrift(x_ref, backend='pytorch', p_val=p_val_threshold)
    preds = cd.predict(x_test)
    
    is_drift = preds['data']['is_drift']
    p_value = preds['data']['p_val']
    distance = preds['data']['distance']
    
    # Score proxy: 1 - p_value (closer to 1 is higher chance of drift)
    score = 1.0 - float(p_value) if p_value is not None else 0.0
    
    return {
        "score": max(0.0, score),
        "is_drift": bool(is_drift),
        "p_value": float(p_value) if p_value is not None else 1.0,
        "distance": float(distance) if distance is not None else 0.0,
        "details": f"MMD p-value: {p_value:.4f}, distance: {distance:.4f}" if p_value is not None else "No p-value calculated"
    }
