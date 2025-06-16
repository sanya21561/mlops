import unittest
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import logging 
from model import ImagePreprocessor, OnnxModelLoader
from pytorch_model import Classifier, BasicBlock

class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        self.test_image_path = "images/n01440764_tench.jpeg"
        logging.getLogger('model').setLevel(logging.CRITICAL)
        logging.getLogger('__main__').setLevel(logging.CRITICAL)

    def tearDown(self):
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)

    def test_preprocess_with_path(self):
        """Test preprocessing with image path."""
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        result = self.preprocessor.preprocess(self.test_image_path)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertTrue(isinstance(result, np.ndarray))
        
    def test_preprocess_with_pil_image(self):
        """Test preprocessing with PIL Image object."""
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        img = Image.open(self.test_image_path)
        result = self.preprocessor.preprocess(img)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertTrue(isinstance(result, np.ndarray))
        
    def test_preprocess_invalid_path(self):
        """Test preprocessing with invalid image path."""
        logging.getLogger('model').setLevel(logging.CRITICAL)
        logging.getLogger('__main__').setLevel(logging.CRITICAL)
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess("nonexistent.jpg")
            
    def test_preprocess_non_rgb(self):
        """Test preprocessing with non-RGB image."""
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        img = Image.open(self.test_image_path).convert("L")  # Convert to grayscale
        result = self.preprocessor.preprocess(img)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertTrue(isinstance(result, np.ndarray))


class TestOnnxModelLoader(unittest.TestCase):
    """Test cases for OnnxModelLoader class."""
    
    def setUp(self):
        self.model_path = "model.onnx"
        self.preprocessor = ImagePreprocessor()
        self.test_image_path = "images/n01440764_tench.jpeg"
        logging.getLogger('model').setLevel(logging.CRITICAL)
        logging.getLogger('__main__').setLevel(logging.CRITICAL)

    def tearDown(self):
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        
    def test_model_loading(self):
        """Test ONNX model loading."""
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        if not Path(self.model_path).exists():
            self.skipTest("ONNX model file not found")
            
        model = OnnxModelLoader(self.model_path)
        self.assertIsNotNone(model.session)
        self.assertIsNotNone(model.input_name)
        self.assertIsNotNone(model.output_name)
        
    def test_model_inference(self):
        """Test model inference."""
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        if not Path(self.model_path).exists():
            self.skipTest("ONNX model file not found")
            
        model = OnnxModelLoader(self.model_path)
        input_array = self.preprocessor.preprocess(self.test_image_path)
        prediction = model.predict(input_array)
        
        self.assertTrue(isinstance(prediction, int))
        self.assertTrue(0 <= prediction < 1000)  # ImageNet has 1000 classes
        
    def test_invalid_input_shape(self):
        """Test inference with invalid input shape."""
        logging.getLogger('model').setLevel(logging.CRITICAL)
        logging.getLogger('__main__').setLevel(logging.CRITICAL)
        if not Path(self.model_path).exists():
            self.skipTest("ONNX model file not found")
            
        model = OnnxModelLoader(self.model_path)
        invalid_input = np.random.rand(1, 3, 100, 100)  # Wrong size
        
        with self.assertRaises(RuntimeError):
            model.predict(invalid_input)


class TestModelConsistency(unittest.TestCase):
    """Test consistency between PyTorch and ONNX models."""
    
    def setUp(self):
        self.weights_path = "weights/pytorch_model_weights.pth"
        self.onnx_path = "model.onnx"
        self.test_image_path = "images/n01440764_tench.jpeg"
        logging.getLogger('model').setLevel(logging.CRITICAL)
        logging.getLogger('__main__').setLevel(logging.CRITICAL)

    def tearDown(self):
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        
    def test_prediction_consistency(self):
        """Test if PyTorch and ONNX models give same predictions."""
        logging.getLogger('model').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        if not all(Path(p).exists() for p in [self.weights_path, self.onnx_path]):
            self.skipTest("Required model files not found")
            
        # Load PyTorch model
        pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
        pytorch_model.load_state_dict(torch.load(self.weights_path))
        pytorch_model.eval()
        
        # Load ONNX model
        onnx_model = OnnxModelLoader(self.onnx_path)
        preprocessor = ImagePreprocessor()
        
        # Get predictions
        img = Image.open(self.test_image_path).convert("RGB")
        input_tensor = pytorch_model.preprocess_numpy(img)
        input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor)
            pytorch_pred = int(torch.argmax(pytorch_output, dim=1)[0])
            
        onnx_input = preprocessor.preprocess(img)
        onnx_pred = onnx_model.predict(onnx_input)
        
        self.assertEqual(pytorch_pred, onnx_pred)


if __name__ == "__main__":
    unittest.main()

