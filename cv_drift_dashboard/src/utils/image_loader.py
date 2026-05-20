import numpy as np
import imagecodecs
from PIL import Image
from io import BytesIO

def load_wmp_image(file_bytes: bytes) -> np.ndarray:
    """
    Decodes Windows Media Photo (.jxr / .wdp) or standard images from bytes.
    Returns a NumPy array representing the image.
    """
    try:
        # Attempt to decode as JPEG XR (WMP)
        img_array = imagecodecs.jpegxr_decode(file_bytes)
        return img_array
    except Exception as e:
        # Fallback to Pillow for standard formats (JPEG, PNG, etc.)
        try:
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
            return np.array(image)
        except Exception as e2:
            raise ValueError(f"Could not decode image format. Errors: JXR({e}), PIL({e2})")

def preprocess_image(img_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Resize image and normalize if required for standard CV models (like ResNet).
    """
    # For now, just resizing. ResNet normalization usually happens in the model pipeline
    # or via torchvision.transforms.
    image = Image.fromarray(img_array).resize(target_size)
    return np.array(image)
