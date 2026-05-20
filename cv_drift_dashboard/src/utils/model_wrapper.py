import torch
import io

class PyTorchModelWrapper:
    """Wrapper to load and run inference on PyTorch models uniformly."""
    
    def __init__(self, model_bytes: bytes):
        buffer = io.BytesIO(model_bytes)
        self.device = torch.device("cpu")
        try:
            # Note: For production, torch.jit.load is safer than torch.load
            # if models are saved via torch.jit.script. 
            self.model = torch.load(buffer, map_location=self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model: {e}")

    def predict(self, input_tensor: torch.Tensor):
        """Run inference on the preprocessed tensor."""
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
        return output
