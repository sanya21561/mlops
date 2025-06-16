import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Union
from PIL import Image
import onnxruntime as ort
# import torch
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for model inference."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor with standard ImageNet normalization.
        
        Args:
            image_size (Tuple[int, int]): Target image size (height, width)
        """
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def preprocess(self, img: Union[str, Image.Image]) -> np.ndarray:
        """
        Preprocess an image for model inference.
        
        Args:
            img: Either a path to an image file or a PIL Image object
            
        Returns:
            np.ndarray: Preprocessed image tensor of shape [1, 3, 224, 224]
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        try:
            # Handle string path or PIL Image
            if isinstance(img, str):
                if not Path(img).exists():
                    raise FileNotFoundError(f"Image file not found: {img}")
                img = Image.open(img)
            
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Apply transformations
            tensor = self.transform(img)  # shape: [3, 224, 224]
            return tensor.unsqueeze(0).numpy()  # shape: [1, 3, 224, 224]
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")


class OnnxModelLoader:
    """Handles ONNX model loading and inference."""
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX model loader.
        
        Args:
            model_path (str): Path to ONNX model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model cannot be loaded
        """
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            logger.info(f"Loading ONNX model from {model_path}")
            self.model_path = model_path
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

    def predict(self, input_array: np.ndarray) -> int:
        """
        Run inference on preprocessed input.
        
        Args:
            input_array (np.ndarray): Preprocessed image tensor of shape [1, 3, 224, 224]
            
        Returns:
            int: Predicted class index
            
        Raises:
            ValueError: If input shape is incorrect
            RuntimeError: If inference fails
        """
        try:
            if input_array.shape != (1, 3, 224, 224):
                raise ValueError(f"Expected input shape (1, 3, 224, 224), got {input_array.shape}")
                
            outputs = self.session.run([self.output_name], {self.input_name: input_array})
            output_array = outputs[0]  # shape: [1, 1000]
            predicted_class = int(np.argmax(output_array, axis=1)[0])
            return predicted_class
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
